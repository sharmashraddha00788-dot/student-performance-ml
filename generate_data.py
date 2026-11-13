# generate_data.py
# This script creates a synthetic student dataset (data/students.csv).
# The data is generated using numpy random numbers with random_state=42.

import os
import numpy as np
import pandas as pd

# Set random seed so the generated data is always the same
np.random.seed(42)

# We start with 490 unique students
n_students = 490

# 1. Generate student ID
student_id = np.arange(1001, 1001 + n_students)

# 2. Generate gender (roughly 52% Male, 48% Female)
gender = np.random.choice(['Male', 'Female'], size=n_students, p=[0.52, 0.48])

# 3. Generate study hours per day (mostly between 1 and 9 hours)
study_hours = np.round(np.random.normal(loc=5.0, scale=2.0, size=n_students), 1)
study_hours = np.clip(study_hours, 0.5, 10.0)

# 4. Generate attendance percentage (between 40% and 100%)
attendance = np.round(np.random.normal(loc=75.0, scale=14.0, size=n_students), 1)
attendance = np.clip(attendance, 40.0, 100.0)

# 5. Generate previous exam marks (between 15 and 98)
previous_marks = np.round(np.random.normal(loc=60.0, scale=18.0, size=n_students), 1)
previous_marks = np.clip(previous_marks, 15.0, 98.0)

# 6. Generate sleep hours per night (between 4 and 9 hours)
sleep_hours = np.random.choice([4, 5, 6, 7, 8, 9], size=n_students, p=[0.05, 0.15, 0.35, 0.30, 0.10, 0.05])

# 7. Generate parental education level
parental_education = np.random.choice(
    ['School', 'Graduate', 'Postgraduate'],
    size=n_students,
    p=[0.30, 0.45, 0.25]
)

# 8. Generate extracurricular activity participation (Yes or No)
extracurricular = np.random.choice(['Yes', 'No'], size=n_students, p=[0.45, 0.55])

# 9. Calculate final marks based on study hours, previous marks, and attendance
# We also add some random Gaussian noise so real-world variance is present
noise = np.random.normal(0, 4.0, size=n_students)
final_marks = 0.40 * previous_marks + 3.2 * study_hours + 0.25 * attendance - 12.0 + noise
final_marks = np.round(np.clip(final_marks, 10.0, 99.0), 1)

# 10. Determine pass or fail: Pass if final_marks >= 35, else Fail
result = np.where(final_marks >= 35.0, 'Pass', 'Fail')

# 11. Assign letter grade based on final marks
def assign_grade(score):
    if score >= 60.0:
        return 'A'
    elif score >= 48.0:
        return 'B'
    elif score >= 35.0:
        return 'C'
    else:
        return 'D'

grade = np.array([assign_grade(m) for m in final_marks])

# Create pandas DataFrame
df = pd.DataFrame({
    'student_id': student_id,
    'gender': gender,
    'study_hours': study_hours,
    'attendance': attendance,
    'previous_marks': previous_marks,
    'sleep_hours': sleep_hours,
    'parental_education': parental_education,
    'extracurricular': extracurricular,
    'final_marks': final_marks,
    'result': result,
    'grade': grade
})

# Introduce a few realistic outliers in study_hours (e.g. students claiming > 15 hours)
outlier_indices = [15, 88, 210, 345]
df.loc[outlier_indices, 'study_hours'] = [15.5, 17.0, 16.2, 18.0]

# Introduce about 5% missing values in study_hours, attendance, and parental_education
missing_study = np.random.choice(df.index.difference(outlier_indices), size=24, replace=False)
df.loc[missing_study, 'study_hours'] = np.nan

missing_att = np.random.choice(df.index, size=24, replace=False)
df.loc[missing_att, 'attendance'] = np.nan

missing_edu = np.random.choice(df.index, size=24, replace=False)
df.loc[missing_edu, 'parental_education'] = np.nan

# Introduce 10 duplicate rows to simulate realistic data entry duplicates
dup_indices = np.random.choice(n_students, size=10, replace=False)
duplicate_rows = df.iloc[dup_indices].copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV file
csv_path = os.path.join('data', 'students.csv')
df.to_csv(csv_path, index=False)

print(f"Dataset successfully generated and saved to {csv_path}")
print(f"Total rows: {len(df)}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Fail percentage: {(df['result'] == 'Fail').mean() * 100:.2f}%")
print(f"Grade distribution:\n{df['grade'].value_counts()}")
