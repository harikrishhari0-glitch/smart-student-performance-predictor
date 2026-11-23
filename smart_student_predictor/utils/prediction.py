FEATURES = [
    "Attendance_Percentage",
    "Study_Hours_Per_Day",
    "Internal_Marks",
    "Assignment_Score",
    "Previous_Semester_Marks",
]


def validate_inputs(attendance, study_hours, internal, assignment, previous, age=None):
    values = {
        "Attendance percentage": (attendance, 0, 100),
        "Study hours per day": (study_hours, 0, 24),
        "Internal marks": (internal, 0, 100),
        "Assignment score": (assignment, 0, 100),
        "Previous semester marks": (previous, 0, 100),
    }
    if age is not None:
        values["Age"] = (age, 15, 100)
    for label, (value, minimum, maximum) in values.items():
        if value is None or not minimum <= float(value) <= maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")


def calculate_support_score(attendance, study_hours, internal, assignment, previous):
    """Returns a 0–100 academic support-strength score for UI guidance."""
    normalized = (
        attendance * 0.20
        + min(study_hours / 8 * 100, 100) * 0.20
        + internal * 0.20
        + assignment * 0.20
        + previous * 0.20
    )
    return round(max(0, min(100, normalized)), 1)


def risk_level(category, attendance, study_hours, final_marks):
    if category == "Needs Improvement" or final_marks < 55:
        return "High Risk"
    if category == "Average" or attendance < 75 or study_hours < 2:
        return "Moderate Risk"
    return "Low Risk"


def generate_recommendations(attendance, study_hours, internal, assignment, previous):
    recommendations = []
    if attendance < 75:
        recommendations.append(("Attendance", "Improve attendance; aim for at least 75% and attend revision classes regularly."))
    if study_hours < 2:
        recommendations.append(("Study Schedule", "Create a consistent daily schedule and gradually study for at least 2–3 hours."))
    elif study_hours < 4:
        recommendations.append(("Study Quality", "Use focused study blocks and reduce distractions during study time."))
    if internal < 60:
        recommendations.append(("Internal Exams", "Revise difficult subjects and seek help from teachers or classmates."))
    if assignment < 60:
        recommendations.append(("Assignments", "Complete assignments regularly and practice questions instead of only reading notes."))
    if previous >= 75 and internal >= 70:
        recommendations.append(("Strengths", "Maintain your current routine and set slightly higher weekly academic targets."))
    if not recommendations:
        recommendations.append(("Overall", "Continue your balanced routine and review weaker topics every week."))
    return recommendations
