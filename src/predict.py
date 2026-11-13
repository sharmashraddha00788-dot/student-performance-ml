# src/predict.py
# This script loads our trained machine learning pipeline and predicts whether
# a student will Pass or Fail based on user-provided inputs.

import os
import joblib
import pandas as pd

def main():
    print("--------------------------------------------------")
    print("Student Performance Prediction Tool")
    print("--------------------------------------------------")

    # Path to the saved model pipeline
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run notebooks/05_pipeline_cv_and_saving.ipynb first.")
        return

    # Load the trained pipeline
    print("Loading model from models/best_model.joblib...")
    pipeline = joblib.load(model_path)
    print("Model loaded successfully.\n")

    print("Please enter the student details:")

    # Get inputs from user
    try:
        study_hours_input = input("Study hours per day (e.g. 5.0): ").strip()
        study_hours = float(study_hours_input) if study_hours_input else 5.0

        attendance_input = input("Attendance percentage (e.g. 75.0): ").strip()
        attendance = float(attendance_input) if attendance_input else 75.0

        previous_marks_input = input("Previous exam marks (e.g. 60.0): ").strip()
        previous_marks = float(previous_marks_input) if previous_marks_input else 60.0

        sleep_hours_input = input("Sleep hours per night (e.g. 7): ").strip()
        sleep_hours = int(sleep_hours_input) if sleep_hours_input else 7

        gender_input = input("Gender (Male/Female) [default: Male]: ").strip().capitalize()
        gender = gender_input if gender_input in ['Male', 'Female'] else 'Male'

        parental_edu_input = input("Parental education (School/Graduate/Postgraduate) [default: Graduate]: ").strip().capitalize()
        if parental_edu_input not in ['School', 'Graduate', 'Postgraduate']:
            parental_education = 'Graduate'
        else:
            parental_education = parental_edu_input

        extracurricular_input = input("Extracurricular activities (Yes/No) [default: No]: ").strip().capitalize()
        extracurricular = 'Yes' if extracurricular_input in ['Yes', 'Y'] else 'No'

    except ValueError as e:
        print(f"Invalid input: {e}. Using default values.")
        study_hours = 5.0
        attendance = 75.0
        previous_marks = 60.0
        sleep_hours = 7
        gender = 'Male'
        parental_education = 'Graduate'
        extracurricular = 'No'

    # Build DataFrame for prediction
    student_data = pd.DataFrame([{
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_marks': previous_marks,
        'sleep_hours': sleep_hours,
        'gender': gender,
        'parental_education': parental_education,
        'extracurricular': extracurricular
    }])

    print("\nStudent Data Summary:")
    print(f"- Study Hours: {study_hours} hrs/day")
    print(f"- Attendance: {attendance}%")
    print(f"- Previous Marks: {previous_marks}")
    print(f"- Sleep Hours: {sleep_hours} hrs/night")
    print(f"- Gender: {gender}")
    print(f"- Parental Education: {parental_education}")
    print(f"- Extracurricular: {extracurricular}")

    # Make prediction using the pipeline
    prediction = pipeline.predict(student_data)[0]
    probabilities = pipeline.predict_proba(student_data)[0]

    # Find probability for predicted class and fail/pass probabilities
    class_probs = dict(zip(pipeline.classes_, probabilities))
    pass_prob = class_probs.get('Pass', 0.0) * 100
    fail_prob = class_probs.get('Fail', 0.0) * 100

    print("\n--------------------------------------------------")
    print(f"Prediction: Student is predicted to {prediction.upper()}")
    print(f"Confidence - Pass: {pass_prob:.2f}%, Fail: {fail_prob:.2f}%")
    print("--------------------------------------------------")

if __name__ == '__main__':
    main()
