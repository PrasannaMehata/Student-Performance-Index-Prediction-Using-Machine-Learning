import streamlit as st
import numpy as np
import joblib

# --- Page Title ---
st.title("🎓 Student Performance Index Predictor")

st.markdown("""
This app predicts a *Student Performance Index* based on input factors:
- Study Hours  
- Papers Practiced  
- Previous Marks  
- Sleeping Hours  
- Extra Participation (encoded)
""")

# --- Input Fields ---
st.sidebar.header("Enter Student Details")

StudyHours = st.sidebar.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
PapersPracticed = st.sidebar.number_input("Papers Practiced per Week", min_value=0, max_value=100, value=10, step=1)
PreviousMarks = st.sidebar.number_input("Previous Marks (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
SleepingHours = st.sidebar.number_input("Average Sleeping Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
ExtraParticipation_encoded = st.sidebar.selectbox(
    "Extra Participation (Encoded)",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# --- Load Saved Model ---
@st.cache_resource
def load_model():
    #Give here the path of your model
    return joblib.load(r"C:\Users\Hp\Downloads\student_performance_index_model.pkl")

model = load_model()

# --- Prepare Input ---
input_features = np.array([[StudyHours, PapersPracticed, PreviousMarks, SleepingHours, ExtraParticipation_encoded]])

# --- Prediction ---
if st.button("Predict Performance Index"):
    try:
        prediction = model.predict(input_features)[0]
        st.success(f"🎯 Predicted Student Performance Index: *{prediction:.2f}*")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.caption("💡 Tip: Ensure your model file student_performance_model.joblib is in the same directory as this app.")
