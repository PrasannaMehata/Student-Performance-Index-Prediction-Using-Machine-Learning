# Student-Performance-Index-Prediction-Using-Machine-Learning

🧮 Student Performance Index Prediction

This project predicts a Student Performance Index (SPI) using a pre-trained Machine Learning model.
It analyzes student behavior and learning parameters such as study hours, participation, sleep patterns, and previous marks to estimate performance levels.

# Note : I Have Already Trained the ML Model 
[Visit my GOOGLE COLAB NOTEBOOK](https://colab.research.google.com/drive/1dIhBx__1yuSJ5Gl0Y_9EgT-y375zarH8?usp=sharing)

🚀 Features

Load and use a pre-trained model with joblib

Predict Performance Index using multiple student attributes

Gradient-colored visualization for performance insights

Simple Streamlit web interface for real-time predictions

Easily extendable to integrate with APIs or databases

🧠 Input Features
Feature	Description
StudyHours	Average study hours per day
ExtraParticipation_encoded	Encoded value for extracurricular participation (1 = Yes, 0 = No)
SleepingHours	Average sleep duration (hours)
PapersPracticed	Number of practice papers attempted
PreviousMarks	Previous academic percentage or GPA

Output	Description:
PerformanceIndex	Predicted performance score (continuous value)

🧩 Dependencies

Install the required Python libraries:

-->pip install pandas numpy matplotlib joblib streamlit


(Optional if you want to use Jupyter)

-->pip install jupyter



⚙️ Usage (Local):------>below

1)Clone the repository:

git clone https://github.com/PrasannaMehata/student-performance-index.git

cd student-performance-index


2)Add your pre-trained model file (e.g., student_performance_model.joblib) to the root directory.

3)Run the prediction script:
python student_performance_predict.py


The script will:--->Load your trained model--->Predict the PerformanceIndex---->Display a gradient-based scatter plot showing study hours vs performance

🌐 Streamlit Web App

You can run an interactive web app for real-time prediction using Streamlit.

Create a file named app.py:
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

4) Run the app:
streamlit run app.py  # (here use --> newpy.py)


Then open the displayed local URL (e.g. http://localhost:8501) in your browser.
