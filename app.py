import streamlit as st
import numpy as np
from PIL import Image
import io
# removed duplicate os import
from utils.model_utils import load_model_safe as load_model_utils, predict_image, get_class_names
from utils.xai_utils import generate_gradcam, create_visualization_plots
from utils.gemini_utils import get_disease_explanation, chat_with_gemini

# fallback plotting for safe wrappers
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Eye Disease Prediction & Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        color: #212529;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .info-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: #212529;
    }
    .info-card h3 { color: #212529; }
    .info-card p { color: #343a40; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

# Header
st.markdown('<h1 class="main-header">👁️ Eye Disease Prediction & Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Retinal Disease Detection with Explainable AI & Medical Insights</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Choose a section:",
        ["🏠 Home", "📤 Upload & Predict", "💬 Chat with AI Doctor", "📊 Disease Statistics", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 🔬 Model Information")
    st.info("Using fine-tuned deep learning model with Grad-CAM visualization")
    
    st.markdown("### 🤖 AI Assistant")
    st.success("Powered by Google Gemini AI")

# Load model
@st.cache_resource
def load_prediction_model():
    """
    Load the prediction model using the central logic in utils.model_utils.

    The underlying loader will look for:
      1. model/fine_tuned_final_model.h5    (primary, user-specified)
      2. models/fine_tuned_final_model.keras (fallback for backward compat)
    """
    # Delegate path resolution to utils.model_utils
    return load_model_utils("models/fine_tuned_final_model.keras") 


try:
    model = load_prediction_model()
    class_names = get_class_names()
except Exception as e:
    st.warning(f"Model not loaded: {e}. Prediction features are disabled.")
    model = None
    class_names = get_class_names()

# -----------------------------
# Safe wrapper helpers (backend only)
# -----------------------------
def safe_generate_gradcam(model, image):
    """Call generate_gradcam but never raise to the Streamlit UI."""
    try:
        return generate_gradcam(model, image)
    except Exception as e:
        # Show a warning but continue with fallbacks
        st.warning(f"Grad-CAM failed: {e}")
        # fallback: return original image as both gradcam_img and heatmap so UI won't break
        return image, image

def safe_create_visualization_plots(all_confidences, prediction):
    """Call create_visualization_plots or return a simple matplotlib bar figure."""
    try:
        return create_visualization_plots(all_confidences, prediction)
    except Exception as e:
        st.warning(f"Visualization creation failed: {e}. Showing simple fallback chart.")
        # Create a simple bar chart as fallback
        fig, ax = plt.subplots()
        labels = list(all_confidences.keys())
        vals = [float(v) for v in all_confidences.values()]
        ax.bar(labels, vals)
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Confidence distribution (fallback)")
        plt.tight_layout()
        return fig

def safe_get_disease_explanation(disease):
    """Call Gemini wrapper but catch errors and return a safe message."""
    try:
        return get_disease_explanation(disease)
    except Exception as e:
        st.warning(f"AI explanation generation failed: {e}")
        return f"⚠️ Unable to generate detailed explanation automatically right now. Detected: **{disease}**. Please consult a medical professional for accurate advice."

# HOME PAGE
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🎯 Accurate Detection</h3>
                <p>State-of-the-art deep learning model trained on thousands of retinal images</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>🔍 Explainable AI</h3>
                <p>Grad-CAM visualizations show exactly what the AI is looking at</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card">
                <h3>🤖 AI Medical Assistant</h3>
                <p>Get detailed explanations, causes, treatments, and prevention tips</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🌟 Key Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        ### 📈 What We Detect
        - **CNV**: Choroidal Neovascularization
        - **DME**: Diabetic Macular Edema
        - **DRUSEN**: Yellow deposits under the retina
        - **NORMAL**: Healthy eye assessment
        """)
        
        st.markdown("""
        ### 🔬 Advanced Analysis
        - Deep learning prediction
        - Confidence scores
        - Heat map visualizations
        - Multi-scale feature analysis
        """)
    
    with features_col2:
        st.markdown("""
        ### 💡 AI Medical Insights
        - Detailed disease explanations
        - Root causes and risk factors
        - Treatment options
        - Prevention strategies
        - Lifestyle recommendations
        """)
        
        st.markdown("""
        ### 🛡️ Eye Health Tips
        - Regular check-ups
        - Protective measures
        - Nutrition guidance
        - Exercise recommendations
        """)

# UPLOAD & PREDICT PAGE
elif page == "📤 Upload & Predict":
    st.markdown("## 📤 Upload Retinal Image for Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a retinal image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear retinal fundus image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # show uploaded image
            st.image(image, caption="Uploaded Image")
            
            if st.button("🔍 Analyze Image", key="predict_btn"):
                if model is None:
                    st.error("Prediction is unavailable because the model is not loaded.")
                    st.stop()
                with st.spinner("🧠 AI is analyzing the image..."):
                    # Prediction
                    prediction, confidence, all_confidences = predict_image(model, image, class_names)
                    st.session_state.current_prediction = prediction
                    
                    # Generate Grad-CAM (safe)
                    gradcam_img, heatmap = safe_generate_gradcam(model, image)
                    
                    # Store in session state
                    st.session_state.prediction_result = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'all_confidences': all_confidences,
                        'gradcam_img': gradcam_img,
                        'heatmap': heatmap,
                        'original_image': image
                    }
                    
                st.success("✅ Analysis Complete!")
                st.rerun()
    
    with col2:
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Diagnosis: {result['prediction']}</h2>
                    <h3>Confidence: {result['confidence']:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence scores
            st.markdown("### 📊 Confidence Distribution")
            for disease, conf in result['all_confidences'].items():
                st.write(f"{disease}: {conf:.2f}%")
                st.progress(conf / 100)
    
    # Visualizations
    if 'prediction_result' in st.session_state:
        st.markdown("---")
        st.markdown("## 🎨 Explainable AI Visualizations")
        
        result = st.session_state.prediction_result
        
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(result['original_image'])
        
        with viz_col2:
            st.markdown("#### 🔥 Grad-CAM Heatmap")
            st.image(result['heatmap'])
        
        with viz_col3:
            st.markdown("#### 🎯 Overlay Visualization")
            st.image(result['gradcam_img'])
        
        st.info("🔍 **Grad-CAM Explanation**: The red areas show regions the AI focused on for diagnosis. Higher intensity indicates stronger influence on the prediction.")
        
        # Additional visualizations
        st.markdown("### 📈 Advanced Analysis")
        viz_plots = safe_create_visualization_plots(result['all_confidences'], result['prediction'])
        st.pyplot(viz_plots)
        
        # Get AI explanation
        st.markdown("---")
        st.markdown("## 🤖 AI Medical Explanation")
        
        with st.spinner("🧠 Generating detailed medical explanation..."):
            explanation = safe_get_disease_explanation(result['prediction'])
            # use markdown but ensure safe display
            st.markdown(explanation)
            
            # Add to chat history
            if explanation not in [msg['content'] for msg in st.session_state.chat_history]:
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': explanation,
                    'disease': result['prediction']
                })

# CHAT WITH AI DOCTOR PAGE
elif page == "💬 Chat with AI Doctor":
    st.markdown("## 💬 Chat with AI Medical Assistant")
    
    st.info("💡 Ask me anything about eye diseases, symptoms, treatments, or prevention!")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
    
    # Chat input
    user_input = st.chat_input("Ask about eye diseases, symptoms, treatments...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get AI response
        with st.spinner("🤖 AI Doctor is thinking..."):
            response = chat_with_gemini(user_input, st.session_state.current_prediction)
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
        
        st.experimental_rerun()
    
    # Quick question buttons
    st.markdown("### 🎯 Quick Questions")
    quick_questions = [
        "What are the symptoms of CNV?",
        "How can I prevent DME if I have diabetes?",
        "What causes DRUSEN?",
        "How to maintain healthy eyes?",
        "What foods are good for eye health?"
    ]
    
    cols = st.columns(3)
    for idx, question in enumerate(quick_questions):
        with cols[idx % 3]:
            if st.button(question, key=f"quick_{idx}"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': question
                })
                response = chat_with_gemini(question, st.session_state.current_prediction)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                st.experimental_rerun()

# STATISTICS PAGE
elif page == "📊 Disease Statistics":
    st.markdown("## 📊 Eye Disease Statistics & Information")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Global Statistics", "🔬 Disease Info", "🎯 Risk Factors", "💊 Treatments"])
    
    with tab1:
        st.markdown("### 🌍 Global Prevalence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("CNV Cases (with AMD)", "11 Million", "Age-related")
            st.metric("DME Cases Worldwide", "21 Million", "Diabetes-related")
        
        with col2:
            st.metric("DRUSEN Prevalence (60+)", "30-40%", "AMD indicator")
            st.metric("Vision Impairment Cases", "2.2 Billion", "Growing")
        
        st.info("📌 Source: WHO Global Report on Vision 2023 & International AMD Studies")
    
    with tab2:
        disease_info = {
            "CNV": {
                "description": "Choroidal Neovascularization - abnormal blood vessel growth beneath the retina",
                "symptoms": ["Distorted vision", "Wavy lines", "Blind spots", "Reduced central vision"],
                "age_group": "Most common in people over 50 with age-related macular degeneration"
            },
            "DME": {
                "description": "Diabetic Macular Edema - fluid buildup in the macula due to diabetes",
                "symptoms": ["Blurred vision", "Wavy vision", "Color changes", "Vision loss"],
                "age_group": "Affects people with diabetes, any age"
            },
            "DRUSEN": {
                "description": "Yellow deposits of protein and lipids under the retina",
                "symptoms": ["Often asymptomatic", "Blurred vision", "Difficulty with low light", "Gradual vision loss"],
                "age_group": "Common in people over 60, sign of AMD risk"
            }
        }
        
        selected_disease = st.selectbox("Select Disease", list(disease_info.keys()))
        info = disease_info[selected_disease]
        
        st.markdown(f"### {selected_disease}")
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Common Age Group:** {info['age_group']}")
        st.write("**Common Symptoms:**")
        for symptom in info['symptoms']:
            st.write(f"- {symptom}")
    
    with tab3:
        st.markdown("### ⚠️ Major Risk Factors")
        
        risk_factors = {
            "Age": "Risk increases significantly after 50",
            "Diabetes": "Major risk for DME and retinal complications",
            "High Blood Pressure": "Increases risk for various conditions",
            "Family History": "Genetic predisposition to AMD and eye diseases",
            "Smoking": "Doubles the risk of AMD and CNV",
            "UV Exposure": "Long-term sun exposure damages retina",
            "Poor Diet": "Lack of antioxidants affects eye health",
            "Obesity": "Linked to diabetes and DME risk"
        }
        
        for factor, description in risk_factors.items():
            st.markdown(f"**{factor}:** {description}")
    
    with tab4:
        st.markdown("### 💊 Common Treatments")
        
        treatments = {
            "CNV": ["Anti-VEGF injections", "Photodynamic therapy", "Laser treatment", "Regular monitoring"],
            "DME": ["Anti-VEGF injections", "Steroid injections", "Laser treatment", "Blood sugar control"],
            "DRUSEN": ["Monitoring", "AREDS vitamins", "Lifestyle changes", "Regular check-ups"]
        }
        
        for disease, treatment_list in treatments.items():
            with st.expander(f"{disease} Treatments"):
                for treatment in treatment_list:
                    st.write(f"✓ {treatment}")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This Eye Disease Prediction & Analysis system combines cutting-edge AI technology with medical expertise to provide:
    
    - **Accurate Disease Detection**: Deep learning model trained on extensive retinal image datasets
    - **Explainable AI**: Grad-CAM visualizations showing what the AI sees
    - **Medical Insights**: Google Gemini AI provides detailed, conversational explanations
    - **Interactive Chat**: Ask questions and get instant medical information
    
    ### 🔬 Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **Deep Learning**: TensorFlow/Keras
    - **Explainable AI**: Grad-CAM implementation
    - **Generative AI**: Google Gemini Pro
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Image Processing**: OpenCV, Pillow
    
    ### 🎓 Model Information
    
    The prediction model is a fine-tuned convolutional neural network trained to detect:
    - CNV (Choroidal Neovascularization)
    - DME (Diabetic Macular Edema)
    - DRUSEN (Yellow deposits under retina)
    - NORMAL (Healthy eyes)
    
    ### ⚠️ Medical Disclaimer
    
    This tool is designed for educational and screening purposes only. It should NOT replace professional medical diagnosis. 
    Always consult with qualified ophthalmologists for accurate diagnosis and treatment.
    
    ### 👥 Contact & Support
    
    For questions, feedback, or support, please reach out through the project repository.
    
    ---
    
    **Made with ❤️ using Streamlit and AI**
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p>👁️ Eye Disease Prediction & Analysis System</p>
        <p>Powered by Deep Learning, Grad-CAM, and Google Gemini AI</p>
        <p style='font-size: 0.8rem;'>⚠️ For educational purposes only. Consult medical professionals for diagnosis.</p>
    </div>
""", unsafe_allow_html=True)
