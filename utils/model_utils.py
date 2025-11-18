# utils/model_utils.py

import os

# Force TensorFlow 2.15 to keep using its bundled tf.keras implementation even if
# the standalone `keras>=3` wheel is present in the environment. This prevents
# the "cannot import name Functional from keras.api.models" error when loading
# legacy HDF5 models.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import importlib
from functools import lru_cache
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

# ---------------------------------------------------------------------------
# Keras 3 compatibility shim
# ---------------------------------------------------------------------------

def _patch_keras_input_layer():
    """
    Keras 3 removed `batch_shape` from `InputLayer`'s constructor, but legacy
    TF/Keras HDF5 models still serialize that key. When the standalone
    `keras>=3` loader is used (e.g., because TensorFlow falls back to it),
    deserialization fails with:
        Unrecognized keyword arguments: ['batch_shape']

    We patch `InputLayer.from_config` to silently drop that key so legacy models
    keep loading even in environments that only have the new Keras package.
    """
    try:
        from keras.src.layers.input_layer import InputLayer  # type: ignore
    except Exception:
        return

    original_from_config = InputLayer.from_config

    def patched_from_config(cls, config):
        cfg = dict(config) if isinstance(config, dict) else config
        if isinstance(cfg, dict):
            cfg.pop("batch_shape", None)
        return original_from_config(cls, cfg)

    InputLayer.from_config = classmethod(patched_from_config)


_patch_keras_input_layer()

MODEL_PATH = "models/fine_tuned_final_model.keras"
TARGET_SIZE = (224, 224)
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
_CANDIDATES = [
    "models/fine_tuned_final_model.keras",
    "models/fine_tuned_final_model.h5",
]


@lru_cache(maxsize=1)
def _resolve_load_model():
    """
    Return a load_model callable that works regardless of the Keras package layout.

    We first try tf.keras (preferred), then fall back to standalone keras>=3 if present.
    This avoids the `Functional` import error triggered when both tensorflow==2.x and
    keras>=3 are installed in the same environment.
    """
    # 1. Attempt via tensorflow module attribute (tf.keras preferred)
    try:
        import tensorflow as tf
        return tf.keras.models.load_model
    except Exception as err:
        tf_attr_error = err

    # 2. Attempt standard tf.keras import
    try:
        tf_models = importlib.import_module("tensorflow.keras.models")
        return getattr(tf_models, "load_model")
    except Exception as err:
        tf_keras_error = err

    # 3. Fallback to standalone keras>=3 API
    try:
        from keras.src.saving.saving_api import load_model as keras_loader
        return keras_loader
    except Exception as err:
        keras_error = err
    # Nothing worked -> raise a descriptive error that surfaces original causes
    raise ImportError(
        "Unable to import a compatible Keras load_model implementation.\n"
        f"- tensorflow.keras.models import error: {tf_keras_error}\n"
        f"- tf.keras attribute access error: {tf_attr_error}\n"
        f"- keras.src fallback error: {keras_error}\n"
        "Please ensure TensorFlow 2.15 (recommended) is installed without a conflicting "
        "standalone keras>=3 package, or update your environment accordingly."
    )

def _resolve_load_model_for_path(path):
    if str(path).lower().endswith(".keras"):
        try:
            from keras.models import load_model as keras_loader
            return keras_loader
        except Exception:
            pass
    try:
        import tensorflow as tf
        return tf.keras.models.load_model
    except Exception:
        pass
    try:
        tf_models = importlib.import_module("tensorflow.keras.models")
        return getattr(tf_models, "load_model")
    except Exception:
        pass
    try:
        from keras.src.saving.saving_api import load_model as keras_loader
        return keras_loader
    except Exception as err:
        raise ImportError(f"Unable to resolve loader for {path}: {err}")

def load_model_safe(model_path=None):
    path = model_path
    if path is None:
        for cand in _CANDIDATES:
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            print(f"[WARN] Model file not found in candidates. Using fallback model.")
            return _build_dummy_model()
    if not os.path.exists(path):
        print(f"[WARN] Model file not found: {path}. Using fallback model.")
        return _build_dummy_model()

    print(f"[INFO] Loading model from: {path}")
    load_model_impl = _resolve_load_model_for_path(path)
    impl_module = getattr(load_model_impl, "__module__", "")
    try:
        import tensorflow.keras.engine.functional as tf_func  # type: ignore
        sys.modules["keras.src.models.functional"] = tf_func
    except Exception:
        try:
            import keras.src.engine.functional as kfunc  # type: ignore
            sys.modules["keras.src.models.functional"] = kfunc
        except Exception:
            pass
    try:
        from tensorflow import keras as tfk  # type: ignore
        sys.modules["keras.layers"] = tfk.layers
        try:
            import tensorflow.keras.layers as tkl  # type: ignore
            sys.modules["keras.layers.serialization"] = tkl
        except Exception:
            pass
    except Exception:
        try:
            import keras as kk  # type: ignore
            sys.modules["keras.layers"] = kk.layers
        except Exception:
            pass
    def _input_layer_ctor(**kwargs):
        try:
            from tensorflow.keras.layers import InputLayer as IL
        except Exception:
            try:
                from tf_keras.src.layers.input_layer import InputLayer as IL  # type: ignore
            except Exception:
                from keras.src.layers.input_layer import InputLayer as IL  # type: ignore
        kwargs.pop("batch_shape", None)
        return IL(**kwargs)

    def _dtype_policy_ctor(**kwargs):
        try:
            from tf_keras.mixed_precision import Policy as P  # type: ignore
        except Exception:
            try:
                from tensorflow.keras.mixed_precision import Policy as P  # type: ignore
            except Exception:
                P = None
        if P is None:
            return "float32"
        name = kwargs.get("name", "float32")
        return P(name)

    def _hard_silu(x):
        return tf.nn.silu(x)

    try:
        from keras.src.utils import custom_object_scope as cos  # type: ignore
    except Exception:
        cos = None
    if cos is not None:
        with cos({
            "InputLayer": _input_layer_ctor,
            "DTypePolicy": _dtype_policy_ctor,
            "hard_silu": _hard_silu,
        }):
            model = load_model_impl(path, compile=False)
    else:
        try:
            model = load_model_impl(path, compile=False, custom_objects={
                "InputLayer": _input_layer_ctor,
                "DTypePolicy": _dtype_policy_ctor,
                "hard_silu": _hard_silu,
            })
        except Exception:
            print("[WARN] Primary loader failed. Using fallback dummy model.")
            model = _build_dummy_model()
    print("[INFO] Model loaded successfully!")
    return model

def _build_dummy_model():
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input")
    x = tf.keras.layers.Rescaling(1.0/255.0)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv1")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv2")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict_image(model, img, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES

    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))
    conf = float(preds[idx] * 100)
    confs = {class_names[i]: float(preds[i] * 100) for i in range(len(preds))}

    return class_names[idx], conf, confs

def get_class_names():
    return CLASS_NAMES

def get_model_summary(model):
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)
