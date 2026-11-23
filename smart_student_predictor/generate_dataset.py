import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def performance_category(score):
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Average"
    return "Needs Improvement"


def generate_dataset(records=600, filename="student_performance_dataset.csv"):
    previous = np.clip(np.random.normal(68, 12, records), 35, 100)
    attendance = np.clip(np.random.normal(82, 12, records), 40, 100)
    study_hours = np.clip(np.random.normal(4.2, 1.8, records), 0.5, 10)
    internal = np.clip(previous * 0.55 + np.random.normal(28, 8, records), 20, 100)
    assignment = np.clip(previous * 0.45 + study_hours * 3.2 + np.random.normal(20, 8, records), 20, 100)

    final_marks = (
        attendance * 0.12
        + study_hours * 1.8
        + internal * 0.27
        + assignment * 0.25
        + previous * 0.19
        + np.random.normal(0, 4.2, records)
    )
    final_marks = np.clip(final_marks, 20, 100)

    student_ids = [f"STU{index:04d}" for index in range(1, records + 1)]
    first_names = ["Aarav", "Aditi", "Diya", "Ishaan", "Kavya", "Krish", "Meera", "Nisha", "Rahul", "Riya", "Sahil", "Tanvi"]
    last_names = ["Sharma", "Verma", "Patel", "Singh", "Kumar", "Gupta", "Mehta", "Joshi", "Iyer", "Reddy"]
    departments = ["BCA", "BCA (AI)", "B.Sc. CS", "B.Com", "BBA"]
    genders = ["Female", "Male", "Prefer not to say"]
    student_names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(records)]
    ages = np.random.randint(17, 23, records)
    selected_genders = np.random.choice(genders, records, p=[0.48, 0.48, 0.04])
    selected_departments = np.random.choice(departments, records)
    semesters = np.random.randint(1, 7, records)
    categories = [performance_category(score) for score in final_marks]

    dataframe = pd.DataFrame({
        "Student_ID": student_ids,
        "Student_Name": student_names,
        "Age": ages,
        "Gender": selected_genders,
        "Department": selected_departments,
        "Semester": semesters,
        "Attendance_Percentage": attendance.round(1),
        "Study_Hours_Per_Day": study_hours.round(1),
        "Internal_Marks": internal.round(1),
        "Assignment_Score": assignment.round(1),
        "Previous_Semester_Marks": previous.round(1),
        "Final_Marks": final_marks.round(1),
        "Performance_Category": categories,
    })
    dataframe.to_csv(filename, index=False)
    print(f"Generated {len(dataframe)} records in {filename}")
    return dataframe


if __name__ == "__main__":
    generate_dataset()
