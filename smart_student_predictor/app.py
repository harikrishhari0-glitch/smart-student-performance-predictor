import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.prediction import FEATURES, calculate_support_score, generate_recommendations, risk_level, validate_inputs

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_performance_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Smart Student Performance Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --primary:#315bea; --navy:#172b4d; --bg:#f4f7fb; --muted:#667085; }
html, body, [class*="css"] {font-family: Inter, "Segoe UI", sans-serif;}
.stApp {background: linear-gradient(145deg,#f8fbff 0%,#f2f6fb 100%);}
.block-container {padding: 1.4rem 2rem 3rem 2rem; max-width: 1450px;}
section[data-testid="stSidebar"] {background:#12233f; border-right:1px solid #1f3560;}
section[data-testid="stSidebar"] * {color:#f7f9fc;}
section[data-testid="stSidebar"] [role="radiogroup"] label {padding:10px 12px; border-radius:10px; margin:3px 0;}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#315bea;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e8edf5; border-radius:16px; padding:18px; box-shadow:0 8px 24px rgba(31,54,92,.07);}
[data-testid="stMetricLabel"] {font-weight:700; color:#667085;}
[data-testid="stMetricValue"] {color:#172b4d; font-size:28px;}
.stDataFrame {border:1px solid #e8edf5; border-radius:14px; overflow:hidden; box-shadow:0 6px 18px rgba(31,54,92,.05);}
.hero {padding:28px 32px; border-radius:22px; background:linear-gradient(135deg,#315bea 0%,#6c63d8 62%,#7d58ba 100%); color:white; box-shadow:0 16px 36px rgba(49,91,234,.20); margin-bottom:20px;}
.hero h1,.hero h2,.hero p {color:white!important;}
.card {background:white; border:1px solid #e8edf5; border-radius:16px; padding:20px; box-shadow:0 8px 24px rgba(31,54,92,.06); height:100%;}
.section-title {color:#172b4d; font-size:1.35rem; font-weight:800; margin:8px 0 14px 0;}
.small-note {color:#667085; font-size:.85rem;}
.predicted-score {font-size:42px; font-weight:850; color:#315bea; line-height:1.1;}
.risk-low {color:#12805c; background:#eaf7f1; border-radius:20px; padding:6px 12px; font-weight:800;}
.risk-moderate {color:#996800; background:#fff6db; border-radius:20px; padding:6px 12px; font-weight:800;}
.risk-high {color:#bb2d3b; background:#fdecec; border-radius:20px; padding:6px 12px; font-weight:800;}
.sidebar-title {font-size:1.05rem; font-weight:850; color:white!important;}
div.stButton > button {border-radius:10px!important; height:48px; border:none; background:#315bea; color:white; font-weight:750; box-shadow:0 8px 18px rgba(49,91,234,.18);}
div.stButton > button:hover {background:#244bd1; color:white; border:none;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


def load_models():
    regressor_path = MODEL_DIR / "performance_regressor.pkl"
    classifier_path = MODEL_DIR / "performance_classifier.pkl"
    metrics_path = MODEL_DIR / "model_metrics.json"
    if not all(path.exists() for path in (regressor_path, classifier_path, metrics_path)):
        return None, None, None
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    return joblib.load(regressor_path), joblib.load(classifier_path), metrics


regressor, classifier, metrics = load_models()
if not DATA_PATH.exists():
    st.error("The dataset is missing. Run: python generate_dataset.py")
    st.stop()
dataframe = load_dataset(DATA_PATH)

with st.sidebar:
    st.markdown('<p class="sidebar-title">🎓 Student Analytics</p>', unsafe_allow_html=True)
    st.caption("Smart Student Performance Predictor")
    page = st.radio("Navigation", ["Home Dashboard", "Student Prediction", "Student Analytics", "Student Records", "Model Information"])
    st.divider()
    st.caption("Project-defined support indicators")

if page == "Home Dashboard":
    st.markdown('<div class="hero"><h2>Welcome to the Smart Student Performance Predictor</h2><p>Explore academic patterns, predict outcomes, and identify students who may benefit from support.</p></div>', unsafe_allow_html=True)
    at_risk = int((dataframe["Performance_Category"] == "Needs Improvement").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{len(dataframe):,}")
    c2.metric("Average Final Marks", f"{dataframe['Final_Marks'].mean():.1f}")
    c3.metric("Average Attendance", f"{dataframe['Attendance_Percentage'].mean():.1f}%")
    c4.metric("At-Risk Students", at_risk)
    st.subheader("Project Overview")
    left, right = st.columns(2)
    with left:
        st.write("This application combines **Random Forest regression** for final-mark estimates with **Random Forest classification** for performance categories.")
        st.write("The predictions help demonstrate early academic support planning; they are not official grading decisions.")
    with right:
        fig = px.histogram(dataframe, x="Final_Marks", nbins=20, color_discrete_sequence=["#2563eb"], title="Final Mark Distribution")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Student Prediction":
    st.markdown("<div class='hero'><h2>Student Performance Prediction</h2><p>Enter the student's profile and academic indicators to generate a detailed support summary.</p></div>", unsafe_allow_html=True)
    if regressor is None:
        st.warning("Models are missing. Run `python generate_dataset.py`, then `python train_model.py`.")
    else:
        with st.form("prediction_form"):
            st.markdown("### 1. Student Details")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                student_name = st.text_input("Student ID or name", "STU-NEW-001")
            with p2:
                age = st.number_input("Age", 15, 100, 20)
            with p3:
                gender = st.selectbox("Gender", ["Female", "Male", "Prefer not to say"])
            with p4:
                department = st.selectbox("Department", ["BCA", "BCA (AI)", "B.Sc. CS", "B.Com", "BBA"])

            c1, c2 = st.columns(2)
            with c1:
                semester = st.number_input("Semester", 1, 8, 4)
                attendance = st.number_input("Attendance percentage", 0.0, 100.0, 75.0, 1.0)
                study_hours = st.number_input("Daily study hours", 0.0, 24.0, 3.0, 0.5)
            with c2:
                internal = st.number_input("Internal marks", 0.0, 100.0, 65.0, 1.0)
                assignment = st.number_input("Assignment score", 0.0, 100.0, 68.0, 1.0)
                previous = st.number_input("Previous semester marks", 0.0, 100.0, 66.0, 1.0)
            submitted = st.form_submit_button("Generate Detailed Prediction", use_container_width=True)

        if submitted:
            try:
                validate_inputs(attendance, study_hours, internal, assignment, previous, age)
                input_frame = pd.DataFrame([{FEATURES[0]: attendance, FEATURES[1]: study_hours, FEATURES[2]: internal, FEATURES[3]: assignment, FEATURES[4]: previous}])
                predicted_marks = float(regressor.predict(input_frame[FEATURES])[0])
                predicted_marks = round(max(0.0, min(100.0, predicted_marks)), 1)
                category = str(classifier.predict(input_frame[FEATURES])[0])
                probabilities = classifier.predict_proba(input_frame[FEATURES])[0]
                confidence = round(float(probabilities.max()) * 100, 1)
                risk = risk_level(category, attendance, study_hours, predicted_marks)
                support_score = calculate_support_score(attendance, study_hours, internal, assignment, previous)

                model_trees = regressor.named_steps["model"]
                transformed_input = regressor.named_steps["preprocessor"].transform(input_frame[FEATURES])
                tree_predictions = [tree.predict(transformed_input)[0] for tree in model_trees.estimators_]
                lower = round(max(0.0, min(100.0, predicted_marks - (2 * pd.Series(tree_predictions).std()))), 1)
                upper = round(max(0.0, min(100.0, predicted_marks + (2 * pd.Series(tree_predictions).std()))), 1)

                a, b, c = st.columns([1.2, 1.2, 1])
                with a:
                    st.markdown("<div class='card'><div class='small-note'>Predicted Final Marks</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='predicted-score'>{predicted_marks:.1f}</div><p>out of 100</p></div>", unsafe_allow_html=True)
                with b:
                    st.markdown("<div class='card'><div class='small-note'>Performance & Risk</div>", unsafe_allow_html=True)
                    st.markdown(f"<h3>{category}</h3><p><span class='risk-{risk.lower().replace(' ', '-').replace('risk','').strip('-')}'>{risk}</span></p></div>", unsafe_allow_html=True)
                with c:
                    st.markdown("<div class='card'><div class='small-note'>Student Details</div>", unsafe_allow_html=True)
                    st.markdown(f"<h3>{student_name}</h3><p>{department} · Semester {semester}<br>{age} years · {gender}</p></div>", unsafe_allow_html=True)

                st.caption(f"Estimated range: {lower:.1f}–{upper:.1f}; category confidence: {confidence:.1f}%")
                st.progress(int(predicted_marks) / 100)

                left, right = st.columns([1.25, 1])
                with left:
                    st.markdown("<p class='section-title'>Academic Indicator Profile</p>", unsafe_allow_html=True)
                    radar = pd.DataFrame({"Indicator": ["Attendance", "Study Hours", "Internal", "Assignments", "Previous"], "Score": [attendance, min(study_hours / 8 * 100, 100), internal, assignment, previous]})
                    fig = go.Figure(go.Scatterpolar(r=radar["Score"].tolist() + [radar["Score"].tolist()[0]], theta=radar["Indicator"].tolist() + [radar["Indicator"].tolist()[0]], fill="toself", line_color="#315bea"))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), height=330, margin=dict(l=30,r=30,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    st.markdown("<p class='section-title'>Support Readiness</p>", unsafe_allow_html=True)
                    gauge = go.Figure(go.Indicator(mode="gauge+number", value=support_score, number={"suffix":" / 100"}, gauge={"axis":{"range":[0,100]}, "bar":{"color":"#315bea"}, "steps":[{"range":[0,55],"color":"#fdecec"},{"range":[55,75],"color":"#fff6db"},{"range":[75,100],"color":"#eaf7f1"}]}))
                    gauge.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(gauge, use_container_width=True)

                st.markdown("<p class='section-title'>Personalized Improvement Plan</p>", unsafe_allow_html=True)
                for title, recommendation in generate_recommendations(attendance, study_hours, internal, assignment, previous):
                    st.write(f"**{title}:** {recommendation}")
                st.caption("These outputs are decision-support indicators based on a synthetic training dataset, not official academic or medical judgments.")
            except ValueError as error:
                st.error(str(error))

elif page == "Student Analytics":
    st.title("Student Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(dataframe, x="Final_Marks", color="Performance_Category", barmode="overlay", title="Marks Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(dataframe, x="Attendance_Percentage", y="Final_Marks", color="Performance_Category", title="Attendance vs Marks"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.scatter(dataframe, x="Study_Hours_Per_Day", y="Final_Marks", title="Study Hours vs Marks"), use_container_width=True)
    with c4:
        st.plotly_chart(px.histogram(dataframe, x="Performance_Category", color="Performance_Category", title="Performance Category Distribution"), use_container_width=True)

    st.subheader("Student Performance Comparison")
    selected = st.multiselect("Select students", dataframe["Student_ID"].tolist(), dataframe["Student_ID"].head(5).tolist())
    comparison = dataframe[dataframe["Student_ID"].isin(selected)]
    if not comparison.empty:
        st.plotly_chart(px.bar(comparison, x="Student_ID", y=["Final_Marks", "Internal_Marks", "Assignment_Score"], barmode="group", title="Selected Student Comparison"), use_container_width=True)

elif page == "Student Records":
    st.markdown("<div class='hero'><h2>Student Records Center</h2><p>Search, filter, compare, upload, and export detailed student records.</p></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded:
        try:
            dataframe = pd.read_csv(uploaded)
            st.success("CSV uploaded successfully.")
        except Exception as error:
            st.error(f"Could not read uploaded CSV: {error}")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        category = st.selectbox("Performance", ["All"] + sorted(dataframe["Performance_Category"].dropna().unique().tolist()) if "Performance_Category" in dataframe else ["All"])
    with c2:
        department = st.selectbox("Department", ["All"] + sorted(dataframe["Department"].dropna().unique().tolist()) if "Department" in dataframe else ["All"])
    with c3:
        search = st.text_input("Search by student ID or name")

    records = dataframe.copy()
    if search:
        search_columns = [column for column in ["Student_ID", "Student_Name"] if column in records]
        if search_columns:
            mask = False
            for column in search_columns:
                mask = mask | records[column].astype(str).str.contains(search, case=False, na=False)
            records = records[mask]
    if category != "All" and "Performance_Category" in records:
        records = records[records["Performance_Category"] == category]
    if department != "All" and "Department" in records:
        records = records[records["Department"] == department]
    if "Final_Marks" in records:
        records = records.sort_values("Final_Marks", ascending=False)

    st.dataframe(records, use_container_width=True, hide_index=True)
    csv = records.to_csv(index=False).encode("utf-8")
    st.download_button("Download records as CSV", csv, "student_records.csv", "text/csv")

elif page == "Model Information":
    st.markdown("<div class='hero'><h2>Model Intelligence Center</h2><p>Transparent results, evaluation metrics, and feature interpretation.</p></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><h3>Random Forest Regressor</h3><p>Predicts the continuous final score by averaging multiple decision trees. Lower MAE and RMSE indicate more accurate predictions; R² indicates the proportion of variance explained.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>Random Forest Classifier</h3><p>Predicts one of four performance categories using a majority of balanced decision trees. Accuracy and balanced accuracy measure test-set classification performance.</p></div>", unsafe_allow_html=True)

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"{metrics['mae']:.2f}")
        c2.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
        c3.metric("R²", f"{metrics['r2_score']:.3f}")
        c4.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        c5.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0) * 100:.2f}%")

        labels = metrics["confusion_labels"]
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
        left, right = st.columns(2)
        with left:
            st.markdown("<p class='section-title'>Confusion Matrix</p>", unsafe_allow_html=True)
            st.dataframe(matrix, use_container_width=True)
            st.caption("Rows show actual categories; columns show predicted categories.")
        with right:
            st.markdown("<p class='section-title'>Regression Feature Importance</p>", unsafe_allow_html=True)
            importance = pd.Series(metrics.get("regressor_feature_importance", {})).sort_values()
            st.plotly_chart(px.bar(importance, orientation="h", labels={"index":"Feature", "value":"Importance"}, color_discrete_sequence=["#315bea"]), use_container_width=True)

        st.markdown("<p class='section-title'>Classification Report</p>", unsafe_allow_html=True)
        report = pd.DataFrame(metrics["classification_report"]).T.round(3)
        st.dataframe(report, use_container_width=True)
    else:
        st.warning("Run `python train_model.py` to calculate actual model metrics.")

    st.markdown("<p class='section-title'>Training and Evaluation</p>", unsafe_allow_html=True)
    st.write("The project uses an 80/20 train-test split with the same test observations for both models. Median imputation and standard scaling are contained inside each pipeline. The test set is not used to train either model. Feature importance is descriptive and does not prove causation.")
