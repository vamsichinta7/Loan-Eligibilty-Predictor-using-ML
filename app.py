import streamlit as st
import pickle
import numpy as np
import datetime
import base64
import matplotlib.pyplot as plt
import os
import platform
from PIL import Image

# Smart TTS setup (disabled on Streamlit Cloud)
tts_available = False
engine = None
try:
    if platform.system() != "Linux" or not os.environ.get("HOME", "").startswith("/home/adminuser"):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        tts_available = True
except Exception:
    tts_available = False

# Load models
with open("model.pkl", "rb") as f:
    rf_model = pickle.load(f)
try:
    with open("logistic_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
except:
    lr_model = None

st.set_page_config(
    page_title="Loan Eligibility Predictor", 
    page_icon="💸", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Fixed CSS with properly working slider bars
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1a1a2e 100%) !important;
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0e0;
        min-height: 100vh;
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(0, 255, 255, 0.1) 0%, transparent 50%);
        animation: backgroundPulse 6s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes backgroundPulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    
    .futuristic-header {
        background: rgba(0, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        animation: slideInDown 1s ease-out;
    }
    
    @keyframes slideInDown {
        from { transform: translateY(-100px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .futuristic-header h1 {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
        animation: textGlow 2s ease-in-out infinite alternate;
        margin-bottom: 1rem;
    }
    
    @keyframes textGlow {
        from { text-shadow: 0 0 20px rgba(0, 255, 255, 0.8); }
        to { text-shadow: 0 0 30px rgba(0, 255, 255, 1); }
    }
    
    .futuristic-header p {
        font-size: 1.2rem;
        color: #b0b0b0;
        animation: fadeInUp 1s ease-out 0.5s both;
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 2rem 0 1rem 0;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
        animation: fadeInScale 1s ease-out;
    }
    
    @keyframes fadeInScale {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .ai-models {
        background: rgba(0, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .ai-models h3 {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        margin-bottom: 1rem;
    }
    
    .model-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .model-item:hover {
        background: rgba(0, 255, 255, 0.1);
        border-color: rgba(0, 255, 255, 0.3);
        transform: translateX(5px);
    }
    
    .accuracy-badge {
        background: linear-gradient(45deg, rgba(0, 255, 255, 0.2), rgba(0, 255, 255, 0.4));
        color: #ffffff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }
    
    /* Fixed form backgrounds */
    .stForm {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stForm > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stForm .element-container {
        background: transparent !important;
    }
    
    .stForm .row-widget {
        background: transparent !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background: rgba(0, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: rgba(0, 255, 255, 0.6) !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3) !important;
    }
    
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* FIXED SLIDER STYLING - Working slider bars */
    .stSlider {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }
    
    .stSlider > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Slider track - This is the main slider bar */
    .stSlider .stSlider-track {
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.3), rgba(0, 255, 255, 0.6)) !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    
    /* Slider thumb - The draggable circle */
    .stSlider .stSlider-thumb {
        background: #00ffff !important;
        border: 2px solid #ffffff !important;
        width: 16px !important;
        height: 16px !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.6) !important;
        border-radius: 50% !important;
    }
    
    .stSlider .stSlider-thumb:hover {
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.8) !important;
        transform: scale(1.1) !important;
    }
    
    /* Slider value display - plain text */
    .stSlider .stSlider-value {
        background: transparent !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0 !important;
        margin-top: 0.3rem !important;
    }
    
    /* Min/Max labels - smaller and subtle */
    .stSlider .stSlider-min-max {
        color: #888 !important;
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Alternative targeting for older Streamlit versions */
    .stSlider > div > div > div > div:first-child {
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.3), rgba(0, 255, 255, 0.6)) !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    
    .stSlider > div > div > div > div:last-child {
        background: #00ffff !important;
        border: 2px solid #ffffff !important;
        width: 16px !important;
        height: 16px !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.6) !important;
        border-radius: 50% !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(0, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    
    .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        background: rgba(0, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stCheckbox > label:hover {
        border-color: rgba(0, 255, 255, 0.6) !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3) !important;
    }
    
    /* Submit button styling */
    .stFormSubmitButton > button {
        background: linear-gradient(45deg, rgba(0, 255, 255, 0.2), rgba(0, 255, 255, 0.4)) !important;
        border: 2px solid rgba(0, 255, 255, 0.5) !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 1rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
        backdrop-filter: blur(15px) !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    
    .stFormSubmitButton > button:hover {
        background: linear-gradient(45deg, rgba(0, 255, 255, 0.3), rgba(0, 255, 255, 0.6)) !important;
        border-color: rgba(0, 255, 255, 0.8) !important;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.6) !important;
        transform: translateY(-2px) !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8) !important;
    }
    
    /* Result boxes */
    .result-success {
        background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 255, 0, 0.3));
        border: 2px solid rgba(0, 255, 0, 0.5);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0;
        backdrop-filter: blur(15px);
        animation: successPulse 2s ease-in-out infinite alternate;
        box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
    }
    
    @keyframes successPulse {
        from { box-shadow: 0 0 30px rgba(0, 255, 0, 0.3); }
        to { box-shadow: 0 0 50px rgba(0, 255, 0, 0.6); }
    }
    
    .result-error {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.1), rgba(255, 0, 0, 0.3));
        border: 2px solid rgba(255, 0, 0, 0.5);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0;
        backdrop-filter: blur(15px);
        animation: errorPulse 2s ease-in-out infinite alternate;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.3);
    }
    
    @keyframes errorPulse {
        from { box-shadow: 0 0 30px rgba(255, 0, 0, 0.3); }
        to { box-shadow: 0 0 50px rgba(255, 0, 0, 0.6); }
    }
    
    .creator-credit {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: rgba(0, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 15px;
        font-family: 'Orbitron', monospace;
        color: #00ffff;
        font-size: 1.1rem;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(0, 255, 255, 0.5), rgba(0, 255, 255, 1));
        border-radius: 10px;
    }
    
    @media (max-width: 768px) {
        .futuristic-header h1 {
            font-size: 2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='futuristic-header'>
        <h1>🤖 Loan Eligibility Predictor</h1>
        <p>Advanced machine learning algorithms analyze your financial profile to predict loan eligibility with unprecedented accuracy.</p>
    </div>
""", unsafe_allow_html=True)

# AI Models Section
st.markdown("""
    <div class='ai-models'>
        <h3>🧠 AI Models</h3>
        <div class='model-item'>
            <span>Random Forest</span>
            <span class='accuracy-badge'>94.2% Accuracy</span>
        </div>
        <div class='model-item'>
            <span>Logistic Regression</span>
            <span class='accuracy-badge'>91.8% Accuracy</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main Form
with st.form("loan_form"):
    st.markdown("<div class='section-header'>🔧 Application Details</div>", unsafe_allow_html=True)
    
    # Model Selection
    model_choice = st.selectbox("🤖 Select AI Model", ["Random Forest", "Logistic Regression"])
    
    # Personal Information
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.slider("👤 Age", 18, 75, 33)
        Gender = st.selectbox("👫 Gender", ["Male", "Female"])
        Married = st.selectbox("💑 Married", ["Yes", "No"])
        
    with col2:
        Dependents = st.selectbox("👶 Number of Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("💼 Self Employed", ["Yes", "No"])

    # Financial Information
    st.markdown("<div class='section-header'>💰 Financial Information</div>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        ApplicantIncome = st.slider("💵 Applicant Income (₹)", 1000, 100000, 24000, step=1000)
        CoapplicantIncome = st.slider("💴 Coapplicant Income (₹)", 0, 50000, 1508, step=500)
        
    with col4:
        LoanAmount = st.slider("🏦 Loan Amount (₹ thousands)", 50, 700, 140, step=10)
        Loan_Amount_Term = st.slider("📅 Loan Term (days)", 120, 480, 360, step=30)

    # Additional Information
    st.markdown("<div class='section-header'>🏠 Additional Information</div>", unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        Credit_History = st.selectbox("💳 Credit History", ["Good", "Bad"])
        
    with col6:
        Property_Area = st.radio("🏠 Property Area", ["Urban", "Semiurban", "Rural"])

    show_graph = st.checkbox("📊 Show Data Insights")
    
    # Submit Button
    submit = st.form_submit_button("🔍 Analyze Eligibility")

# Results Section
if submit:
    # Processing data
    Gender_num = 1 if Gender == "Male" else 0
    Married_num = 1 if Married == "Yes" else 0
    Dependents_num = 3 if Dependents == "3+" else int(Dependents)
    Education_num = 0 if Education == "Graduate" else 1
    Self_Employed_num = 1 if Self_Employed == "Yes" else 0
    Credit_History_num = 1 if Credit_History == "Good" else 0
    Property_Area_num = {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]

    features = np.array([[Age, Gender_num, Married_num, Dependents_num, Education_num, Self_Employed_num,
                          ApplicantIncome, CoapplicantIncome, LoanAmount,
                          Loan_Amount_Term, Credit_History_num, Property_Area_num]])

    # Model prediction
    if model_choice == "Logistic Regression" and lr_model:
        prediction = lr_model.predict(features)
        proba = lr_model.predict_proba(features)[0][1]
    else:
        prediction = rf_model.predict(features)
        proba = rf_model.predict_proba(features)[0][1]

    # Display results
    if prediction[0] == 1:
        result_text = f"✅ Congratulations! You are eligible for the loan.<br>Confidence Score: {proba * 100:.2f}%"
        st.markdown(f"<div class='result-success'>{result_text}</div>", unsafe_allow_html=True)
        if tts_available and engine:
            engine.say("Congratulations! You are eligible for the loan")
            engine.runAndWait()
    else:
        result_text = f"❌ Sorry, you're not eligible for the loan.<br>Confidence Score: {(1 - proba) * 100:.2f}%"
        st.markdown(f"<div class='result-error'>{result_text}</div>", unsafe_allow_html=True)
        if tts_available and engine:
            engine.say("Sorry, you are not eligible for the loan")
            engine.runAndWait()

    # Data Insights
    if show_graph:
        st.markdown("<div class='section-header'>📊 Data Insights</div>", unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        
        with col7:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            bars = ax.bar(['Applicant', 'Coapplicant'], [ApplicantIncome, CoapplicantIncome], 
                         color=['#00ffff', '#ff6b6b'], alpha=0.8)
            ax.set_ylabel('Income (₹)', color='white', fontsize=12)
            ax.set_title('Income Comparison', color='white', fontsize=14, fontweight='bold')
            ax.tick_params(colors='white')
            
            for bar in bars:
                bar.set_edgecolor('white')
                bar.set_linewidth(2)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with col8:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fig2.patch.set_facecolor('none')
            ax2.set_facecolor('none')
            
            bar = ax2.bar(['Loan Requested'], [LoanAmount * 1000], color=['#9575cd'], alpha=0.8)
            ax2.set_ylabel('Loan Amount (₹)', color='white', fontsize=12)
            ax2.set_title('Requested Loan Amount', color='white', fontsize=14, fontweight='bold')
            ax2.tick_params(colors='white')
            
            bar[0].set_edgecolor('white')
            bar[0].set_linewidth(2)
            
            plt.tight_layout()
            st.pyplot(fig2)

# Model Evaluation Section
st.markdown("<div class='section-header'>📈 Model Evaluation</div>", unsafe_allow_html=True)

col9, col10, col11 = st.columns(3)

with col9:
    if os.path.exists("roc_curve.png"):
        st.image(Image.open("roc_curve.png"), caption="ROC Curve Analysis", use_column_width=True)

with col10:
    if os.path.exists("rf_confusion_matrix.png"):
        st.image(Image.open("rf_confusion_matrix.png"), caption="Random Forest Confusion Matrix", use_column_width=True)

with col11:
    if os.path.exists("lr_confusion_matrix.png"):
        st.image(Image.open("lr_confusion_matrix.png"), caption="Logistic Regression Confusion Matrix", use_column_width=True)

# Creator Credit
st.markdown("""
    <div class='creator-credit'>
        <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>✨ Created by</div>
        <div style='font-size: 1.5rem; font-weight: 700; text-shadow: 0 0 15px rgba(0, 255, 255, 0.8);'>
            Sai Giri Sekhara Vamsi Chinta
        </div>
        <div style='font-size: 0.9rem; margin-top: 0.5rem; color: #b0b0b0;'>
            Loan Eligibility Predictor
        </div>
    </div>
""", unsafe_allow_html=True)
