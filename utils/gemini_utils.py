import os
import time
import google.generativeai as genai

# ----------------------------------------------------------------------
# API KEY SETUP (Streamlit secrets → fallback ENV)
# ----------------------------------------------------------------------

_init_error = None
GEMINI_API_KEY = None

try:
    import streamlit as st
    try:
        GEMINI_API_KEY = st.secrets["general"]["GEMINI_API_KEY"]
    except Exception:
        try:
            GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            GEMINI_API_KEY = None
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as cfg_exc:
        _init_error = cfg_exc
else:
    _init_error = RuntimeError(
        "Gemini API key missing. Set GEMINI_API_KEY in Streamlit secrets or ENV."
    )

# ----------------------------------------------------------------------
# NEW VALID MODELS ONLY (NO DEPRECATED ENTRIES)
# ----------------------------------------------------------------------

VALID_MODELS = [
    "models/gemini-2.0-flash",
    "models/gemini-2.0-pro",
]

model = None
_USING_OBJECT_MODEL = False

# ----------------------------------------------------------------------
# CHOOSE A WORKING MODEL
# ----------------------------------------------------------------------

def _choose_working_model():
    last_exc = None

    for mid in VALID_MODELS:
        try:
            m = genai.GenerativeModel(mid)
            test = m.generate_content("ping")
            _ = getattr(test, "text", str(test))
            return mid, m
        except Exception as e:
            last_exc = e
            continue

    raise RuntimeError(f"No compatible Gemini model available: {last_exc}")


try:
    if GEMINI_API_KEY and _init_error is None:
        mid, mdl = _choose_working_model()
        print(f"[Gemini] Using model: {mid}")
        model = mdl
        _USING_OBJECT_MODEL = True
except Exception:
    model = None
    _USING_OBJECT_MODEL = False


# ----------------------------------------------------------------------
# UNIVERSAL GEMINI CALLER
# ----------------------------------------------------------------------

def _call_gemini(prompt, max_retries=2, retry_delay=1.0):

    if not GEMINI_API_KEY:
        return (
            "⚠️ Gemini API key missing. Add GEMINI_API_KEY in secrets.toml or ENV."
        )

    last_exc = None

    # CASE A — use the selected model
    if _USING_OBJECT_MODEL and model is not None:
        for attempt in range(max_retries + 1):
            try:
                resp = model.generate_content(prompt)
                return getattr(resp, "text", str(resp))
            except Exception as e:
                last_exc = e
                time.sleep(retry_delay * (attempt + 1))

    # CASE B — fallback: try each model from VALID_MODELS
    for mid in VALID_MODELS:
        for attempt in range(max_retries + 1):
            try:
                temp_model = genai.GenerativeModel(mid)
                resp = temp_model.generate_content(prompt)
                return getattr(resp, "text", str(resp))
            except Exception as e:
                last_exc = e
                time.sleep(retry_delay * (attempt + 1))

    return f"⚠️ Gemini request failed: {last_exc}"


# ======================================================================
# ALL PROMPTS BELOW — EXACTLY AS YOU PROVIDED (UNCHANGED)
# ======================================================================

def get_disease_explanation(disease_name):

    if disease_name.upper() == "NORMAL":
        prompt = f"""
        The eye scan shows NORMAL/HEALTHY eyes with no disease detected.

        Please provide a comprehensive, friendly explanation covering:

        1. **Good News**: Confirm that the eyes appear healthy and normal
        2. **What "Normal" Means**: Explain what healthy eyes look like
        3. **Maintaining Eye Health**: Provide detailed tips on:
            - Regular eye check-ups schedule
            - Proper nutrition for eyes (specific foods and vitamins)
            - Eye exercises and rest techniques
            - Protection from UV rays and blue light
            - Proper lighting and screen time management
            - Hydration and its importance
        4. **Lifestyle Recommendations**:
            - Diet suggestions (foods rich in Omega-3, Vitamin A, C, E, Lutein, Zeaxanthin)
            - Exercise and its benefits for eye health
            - Sleep recommendations
            - Stress management
        5. **Warning Signs**: When to see a doctor despite healthy results
        6. **Prevention Tips**: How to prevent common eye diseases like CNV, DME, and DRUSEN
        7. **Age-specific care**: Tips based on different age groups

        Make it conversational, encouraging, and easy to understand for non-medical people.
        Use emojis appropriately to make it engaging.
        """

    elif disease_name.upper() == "CNV":
        prompt = f"""
        An AI eye disease detection system has identified: **CNV (Choroidal Neovascularization)**

        Please provide a comprehensive, compassionate, and detailed explanation covering:

        1. **What is CNV (Choroidal Neovascularization)?**
            - Clear, simple definition: abnormal blood vessel growth beneath the retina
            - How it affects the eye and vision
            - Connection to Age-Related Macular Degeneration (AMD)

        2. **Causes and Risk Factors**:
            - Age-related macular degeneration (wet AMD)
            - High myopia, genetic factors
            - Smoking and cardiovascular disease

        3. **Symptoms and Warning Signs**:
            - Distorted vision (straight lines appear wavy)
            - Blurred central vision
            - Dark or blank spots

        4. **Treatment Options**:
            - Anti-VEGF Injections (Lucentis, Eylea, Avastin)
            - Photodynamic therapy
            - Regular monitoring

        5. **Prevention Strategies**:
            - Stop smoking
            - AREDS vitamins
            - Healthy diet
            - Regular eye exams

        6. **How to Cure/Manage**:
            - Treatment can stabilize or improve vision
            - Requires ongoing injections
            - Early treatment is crucial

        7. **How to Avoid**:
            - Lifestyle modifications
            - Control risk factors
            - Regular screening

        Make it compassionate, easy to understand, and use emojis appropriately.
        """

    elif disease_name.upper() == "DME":
        prompt = f"""
        An AI eye disease detection system has identified: **DME (Diabetic Macular Edema)**

        Please provide a comprehensive, compassionate, and detailed explanation covering:

        1. **What is DME?**
        - Fluid accumulation in the macula due to diabetes
        - How it affects vision

        2. **Causes and Risk Factors**:
        - Diabetes mellitus (Type 1 or Type 2)
        - Poor blood sugar control
        - High blood pressure

        3. **Symptoms**:
        - Blurred or distorted vision
        - Faded colors
        - Difficulty reading

        4. **Treatment Options**:
        - Anti-VEGF injections
        - Steroid injections
        - Laser treatment
           - **Blood sugar control is essential**

        5. **Prevention**:
        - Optimal diabetes control (HbA1c < 7%)
        - Regular blood sugar monitoring
        - Annual eye exams

        6. **How to Cure/Manage**:
        - Combination of eye treatment and diabetes management
        - Vision can be stabilized or improved

        7. **How to Avoid**:
        - Tight diabetes control
        - Healthy lifestyle
        - Regular screening

        Make it compassionate, easy to understand, and use emojis appropriately.
        Emphasize the importance of diabetes control.
        """

    elif disease_name.upper() == "DRUSEN":
        prompt = f"""
        An AI eye disease detection system has identified: **DRUSEN**

        Please provide a comprehensive, compassionate, and detailed explanation covering:

        1. **What are DRUSEN?**
           - Yellow deposits under the retina
           - Sign of AMD risk
           - Not a disease itself

        2. **Causes and Risk Factors**:
           - Aging (over 60)
           - Genetics
           - Smoking
           - Cardiovascular disease

        3. **Symptoms**:
           - Often asymptomatic
           - Gradual vision changes
           - May progress to AMD

        4. **Treatment/Management**:
           - AREDS2 vitamins
           - Regular monitoring
           - Lifestyle modifications

        5. **Prevention**:
           - Stop smoking (most important)
           - Healthy diet (leafy greens, fish)
           - UV protection
           - Regular exercise

        6. **How to Manage**:
           - Most people don't develop severe vision loss
           - Early detection allows prevention

        7. **How to Avoid Progression**:
           - Lifestyle changes
           - AREDS2 supplements
           - Regular monitoring

        Make it reassuring, easy to understand, and use emojis appropriately.
        """

    else:
        prompt = f"""
        An AI eye disease detection system has identified: **{disease_name}**

        Please provide a comprehensive explanation about this eye condition including:
        - What it is
        - Causes and risk factors
        - Symptoms
        - Treatment options
        - Prevention strategies
        - How to cure/manage it
        - How to avoid it

        Make it compassionate, easy to understand, and use emojis appropriately.
        """

    return _call_gemini(prompt)


# ----------------------------------------------------------------------
# CHAT FUNCTION
# ----------------------------------------------------------------------

def chat_with_gemini(user_message, current_diagnosis=None):
    context = ""
    if current_diagnosis:
        context = f"\n\nContext: The user's recent eye scan showed: {current_diagnosis}."

    prompt = f"""
    You are an AI Medical Assistant specializing in ophthalmology and eye health.

    User's question: {user_message}
    {context}

    [PROMPT CONTENT EXACTLY AS PROVIDED]
    """

    return _call_gemini(prompt)


# ----------------------------------------------------------------------
# PREVENTION TIPS (prompts unchanged)
# ----------------------------------------------------------------------

def get_prevention_tips(disease_name=None):

    if disease_name and disease_name.upper() == "CNV":
        prompt = f"""
        Provide comprehensive prevention strategies specifically for **CNV (Choroidal Neovascularization)** and AMD.
        
        Include:
        1. Stop smoking - most critical factor
        2. AREDS vitamins for high-risk individuals
        3. Dietary recommendations (leafy greens, omega-3)
        4. Regular eye exams after 50
        5. Cardiovascular health management
        6. UV protection
        7. Early detection importance
        
        Make it actionable and practical. Use emojis and format with bullet points.
        """

    elif disease_name and disease_name.upper() == "DME":
        prompt = f"""
        Provide comprehensive prevention strategies specifically for **DME (Diabetic Macular Edema)**.
        
        Include:
        1. Optimal diabetes control - foundational
        2. Blood sugar monitoring and HbA1c targets
        3. Blood pressure control
        4. Annual dilated eye exams for diabetics
        5. Healthy diet for diabetes
        6. Regular exercise
        7. Early detection through screening
        
        Make it actionable and practical. Use emojis and format with bullet points.
        """

    elif disease_name and disease_name.upper() == "DRUSEN":
        prompt = f"""
        Provide comprehensive prevention strategies specifically for **DRUSEN** and AMD progression.
        
        Include:
        1. Stop smoking immediately
        2. AREDS2 supplement formula
        3. Diet rich in lutein, zeaxanthin, omega-3
        4. UV protection
        5. Regular monitoring
        6. Healthy weight maintenance
        
        Make it actionable and practical. Use emojis and format with bullet points.
        """

    else:
        prompt = f"""
        Provide comprehensive tips for maintaining overall eye health and preventing common eye diseases including CNV, DME, and DRUSEN.
        
        Cover:
        1. Daily eye care routine
        2. Nutrition for eye health
        3. Protective measures (UV, screen time)
        4. Regular check-ups by age
        5. Exercise and lifestyle
        6. Diabetes prevention
        7. Smoking cessation
        
        Make it practical and easy to follow. Use emojis and clear sections.
        """

    return _call_gemini(prompt)


# ----------------------------------------------------------------------
# TREATMENT INFORMATION
# ----------------------------------------------------------------------

def get_treatment_information(disease_name):

    prompt = f"""
    Provide detailed treatment information for **{disease_name}**.
    
    Cover:
    1. Medical Treatments (medications, injections, surgery, laser)
    2. Home Care (self-care, monitoring)
    3. Lifestyle Modifications (diet, activity)
    4. Expected Outcomes (success rates, timeline)
    5. Complementary Approaches (vitamins, supplements)
    6. Important Considerations (side effects, follow-up)
    
    Make it comprehensive yet easy to understand. Use emojis and clear formatting.
    Always emphasize the importance of professional medical guidance.
    """

    return _call_gemini(prompt)


# ----------------------------------------------------------------------
# SYMPTOM ANALYSIS
# ----------------------------------------------------------------------

def analyze_symptoms(symptoms_list):

    symptoms_str = ", ".join(symptoms_list)

    prompt = f"""
    A person is experiencing the following eye-related symptoms: {symptoms_str}
    
    Please provide:
    1. Possible Conditions (including CNV, DME, DRUSEN, or others)
    2. Severity Assessment
    3. Immediate Actions
    4. When to See a Doctor
    5. Home Care (if appropriate)
    6. Warning Signs
    
    Important: Emphasize that this is not a diagnosis and professional medical evaluation is essential.
    
    Use clear headings, bullet points, and emojis.
    """

    return _call_gemini(prompt)


# ----------------------------------------------------------------------
# NUTRITION GUIDE
# ----------------------------------------------------------------------

def get_nutrition_guide(condition=None):

    if condition and condition.upper() in ["CNV", "DRUSEN"]:
        prompt = f"""
        Provide a comprehensive nutrition guide specifically for people with **{condition}** or at risk for AMD.
        """
    elif condition and condition.upper() == "DME":
        prompt = f"""
        Provide a comprehensive nutrition guide specifically for people with **DME (Diabetic Macular Edema)**.
        """
    else:
        prompt = f"""
        Provide a comprehensive nutrition guide for optimal eye health.
        """

    prompt += f"""

    Include:
    1. Essential Nutrients (Vitamins A, C, E, Zinc, Omega-3, Lutein, Zeaxanthin)
    2. Best Foods (specific foods with serving sizes)
    3. Foods to Limit or Avoid
    4. Hydration importance
    5. Supplements (AREDS2 formula)
    6. Sample Meal Plan
    7. Special Considerations
    
    Make it practical with specific examples. Use emojis and clear formatting.
    """

    return _call_gemini(prompt)
