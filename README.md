
👁️ Eye Disease Prediction & Analysis System

An advanced AI-powered web application for detecting and analyzing retinal diseases using deep learning, Grad-CAM explainability, and an intelligent medical assistant powered by Google Gemini AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.12.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

🌟 Features

🎯 Disease Detection

✔ Detects **CNV**, **DME**, **DRUSEN**, and **NORMAL**
✔ Uses a **fine-tuned MobileNetV3-Small** model
✔ Provides **confidence scores** and top-K predictions
✔ Fast real-time inference

 Explainable AI (XAI)

✔ Grad-CAM heatmaps
✔ Overlay visualizations
✔ Interpretable model decisions
✔ Confidence distribution charts

🤖 AI Medical Assistant (Gemini Pro)

✔ Disease explanations
✔ Causes & symptoms
✔ Prevention & treatment info
✔ Context-aware chat
✔ Lifestyle and eye-care recommendations

📊 Analytics

✔ Confidence charts (bar, pie, ranked)
✔ Disease summaries
✔ Global statistics (informational)

💎 Modern UI

✔ Fully responsive Streamlit UI
✔ Smooth gradients and clean layout
✔ Multi-page navigation
✔ Interactive chat module


📁 Project Structure


eye-disease-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── fine_tuned_final_model.keras
│
├── utils/
│   ├── model_utils.py
│   ├── xai_utils.py
│   └── gemini_utils.py
│
└── sample_images/
    ├── cnv/
    ├── dme/
    ├── drusen/
    └── normal/


🚀 Installation

1️⃣ Clone repository

```bash
git clone https://github.com/keerthika1307ak/eye-disease-prediction.git
cd eye-disease-prediction
```

2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add model

Place your trained model here:

```
models/fine_tuned_final_model.keras
```
 🎮 Usage

Run the application:

```bash
streamlit run app.py
```

Open in browser:
👉 [http://localhost:8501](http://localhost:8501)

In the App:

📤 Upload & Predict**

* Upload OCT retinal scan
* View prediction + confidence
* Grad-CAM heatmap
* Gemini AI explanation

💬 Chat with AI Doctor**

* Ask about symptoms, diseases, treatments
* Context-aware responses using your detected result

📊 Statistics**

* Learn about diseases
* Understand risk factors & treatments

---

🔬 Supported Diseases

| Disease    | Description                                  |
| ---------- | -------------------------------------------- |
| CNV**    | Abnormal blood vessel growth under retina    |
| DME**    | Fluid accumulation in macula due to diabetes |
| DRUSEN   | Yellow deposits under retina (AMD indicator) |
| NORMAL   | Healthy eye                                  |

---

🔧 Configuration

Class Names

```python
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
```

Gemini API Key

In `.streamlit/secrets.toml`:

```toml
[general]
GEMINI_API_KEY="your_key_here"
```

---

📊 Model Information

* Architecture: **MobileNetV3-Small (fine-tuned)**
* Input size: **224 × 224**
* Parameters: Lightweight & optimized
* Best for: Real-time OCT image classification

---

🎨 Customization

Modify UI styling in `app.py`:

```python
st.markdown("""
    <style>
    /* custom CSS */
    </style>
""", unsafe_allow_html=True)
```


❗ Medical Disclaimer

This tool is for **educational and early screening** purposes only.
It does **not** replace professional medical diagnosis.


🐛 Troubleshooting

❌ *Model Not Loaded*

> “Could not locate class 'Functional'”
> ✔ Fixed by Keras alias patching inside `model_utils.py`.

❌ *Gemini API Error*

✔ Ensure valid API key in secrets
✔ Update `google-generativeai` package

❌ *Grad-CAM Error*

✔ Model must have at least one Conv layer
✔ Use updated `xai_utils.py`


🤝 Contributing

Pull requests are welcome!

📄 License

Licensed under the **MIT License**.

👥 Author

**Keerthika Anandhan**
AI & Deep Learning Enthusiast 👩‍💻


🗺️ Roadmap

* [ ] Add more diseases
* [ ] Multi-language support
* [ ] Mobile app version
* [ ] PDF medical report generation
* [ ] Doctor dashboard

"# Eye_disesase_prediction" 
"# Eye_disesase_prediction" 
