import tensorflow as tf
try:
    # Prefer TensorFlow's bundled Keras
    from tensorflow import keras as tf_keras
except Exception:
    tf_keras = None
try:
    # Standalone keras (v3+) as a fallback only
    import keras as keras_api
except Exception:
    keras_api = None
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm  # for colormaps
from matplotlib.patches import Rectangle

# =======================
#  GRAD-CAM CORE LOGIC
# =======================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap (cv2-free version)
    """
    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break

    backend = keras_api or tf_keras
    try:
        grad_model = backend.models.Model(
            inputs=[model.input],
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
    except:
        # fallback: pick any conv layer
        conv_layers = [L for L in model.layers if "conv" in L.name.lower()]
        if not conv_layers:
            return np.random.rand(img_array.shape[1], img_array.shape[2])
        last_conv_layer_name = conv_layers[-1].name
        grad_model = backend.models.Model(
            inputs=[model.input],
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # Weight feature maps
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


# ==========================
#  HEATMAP OVERLAY (NO CV2)
# ==========================

def apply_heatmap_overlay(original_image, heatmap, alpha=0.4):
    """
    Overlay heatmap on image using Matplotlib colormap (cv2-free)
    """
    if isinstance(original_image, Image.Image):
        img = np.array(original_image)
    else:
        img = original_image

    # Resize heatmap to match image
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype("uint8")).resize(
            (img.shape[1], img.shape[0])
        )
    )

    # Use matplotlib cmap
    colormap = cm.get_cmap("jet")
    heatmap_color = colormap(heatmap_resized / 255.0)[:, :, :3]  # ignore alpha
    heatmap_color = (heatmap_color * 255).astype("uint8")

    # Blend
    overlay = (img * (1 - alpha) + heatmap_color * alpha).astype("uint8")

    return Image.fromarray(overlay)


# ===============================
#  COMPLETE GRAD-CAM GENERATOR
# ===============================

def generate_gradcam(model, image, target_size=(224, 224)):
    """
    Fast GradCAM generator without cv2
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = image.resize(target_size)
    img_array = np.array(resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap_img = Image.fromarray(
        (cm.get_cmap("jet")(heatmap)[:, :, :3] * 255).astype("uint8")
    )

    overlay = apply_heatmap_overlay(resized, heatmap)

    return overlay, heatmap_img


# ===============================
#  CONFIDENCE VISUALIZATIONS
# ===============================

def create_visualization_plots(confidence_dict, predicted_class):
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    fig = plt.figure(figsize=(15, 5))

    # Bar Chart
    ax1 = plt.subplot(1, 3, 1)
    classes = list(confidence_dict.keys())
    confidences = list(confidence_dict.values())
    colors = ['#ff6b6b' if c == predicted_class else '#4ecdc4' for c in classes]

    bars = ax1.barh(classes, confidences, color=colors)
    ax1.set_title("Prediction Confidence")

    for i, bar in enumerate(bars):
        ax1.text(confidences[i] + 2, i, f"{confidences[i]:.1f}%", va="center")

    # Pie Chart
    ax2 = plt.subplot(1, 3, 2)
    explode = [0.1 if c == predicted_class else 0 for c in classes]
    ax2.pie(confidences, labels=classes, autopct="%1.1f%%", explode=explode)
    ax2.set_title("Confidence Distribution")

    # Sorted Bar Chart
    ax3 = plt.subplot(1, 3, 3)
    sorted_items = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    cols = ['#ff6b6b' if l == predicted_class else '#95e1d3' for l in labels]

    ax3.bar(labels, values, color=cols)
    ax3.set_title("Ranked Confidence")
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig
def _ensure_keras_aliases():
    try:
        # Always prefer tf.keras for model/graph introspection
        target = tf_keras if tf_keras is not None else keras_api
        if target is None:
            return
        mapping = {
            "keras.src": target,
            "keras.src.models": target.models,
            "keras.src.models.functional": target.models,
            "keras.src.layers": target.layers,
            "keras.src.engine": getattr(target, "engine", target),
            "keras.src.saving": getattr(target, "saving", target),
            "keras.api": target,
            "keras.api.models": target.models,
            "keras.api.layers": target.layers,
            "keras.api.initializers": target.initializers,
            "keras.api.saving": getattr(target, "saving", target),
            "keras.api._v2": target,
            "keras.api._v2.keras": target,
            "keras.api._v2.keras.models": target.models,
            "keras.api._v2.keras.layers": target.layers,
            "keras.api._v2.keras.initializers": target.initializers,
            "keras.api._v2.keras.saving": getattr(target, "saving", target),
        }
        for alias, real in mapping.items():
            sys.modules[alias] = real
    except Exception:
        pass

_ensure_keras_aliases()
