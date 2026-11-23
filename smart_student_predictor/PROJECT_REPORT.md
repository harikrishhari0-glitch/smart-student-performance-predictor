# Smart Student Performance Predictor Using Machine Learning

## Cover Page
**Project Title:** Smart Student Performance Predictor Using Machine Learning

**Academic Program:** BCA – Artificial Intelligence

**Submitted by:** Student Group Members

**Academic Year:** 2026–2027

## Abstract
This project predicts expected final student marks and performance categories using academic and behavioral indicators. It uses a Random Forest Regressor for score prediction and a Random Forest Classifier for category prediction. The Streamlit application provides a dashboard, interactive records, analytics, and study recommendations. It is intended as an educational decision-support project, not an official academic evaluation system.

## 1. Introduction
The project examines attendance, daily study time, internal marks, assignment scores, and previous semester marks. It demonstrates a complete data science workflow from dataset generation to model deployment.

## 2. Problem Statement
Teachers and institutions need early indications of students who may require support. Individual indicators can be difficult to interpret together. This application combines these indicators using machine learning and presents understandable results.

## 3. Objectives
- Generate realistic student data.
- Predict final marks using regression.
- Classify students into four performance levels.
- Identify possible academic risk.
- Provide rule-based improvement suggestions.
- Present results through an interactive dashboard.

## 4. Scope
The system uses project-generated or uploaded CSV data. It supports prediction, record management, and analytics for educational demonstration purposes.

## 5. Methodology
1. Generate 600 synthetic records.
2. Perform a stratified-independent 80/20 train-test split.
3. Preprocess features using median imputation and standard scaling.
4. Train Random Forest models.
5. Calculate MAE, R², accuracy, and confusion matrix.
6. Deploy predictions in a Streamlit interface.

## 6. Technologies Used
Python 3, Pandas, NumPy, scikit-learn, Matplotlib, Plotly, Streamlit, and Joblib.

## 7. System Architecture
CSV dataset → preprocessing pipeline → Random Forest training → serialized models → prediction utility → Streamlit interface.

## 8. Machine Learning Algorithms
A Random Forest Regressor uses many decision trees and averages their outputs to predict continuous final marks. A Random Forest Classifier uses many trees and majority voting to classify performance categories. Random Forest was selected because it models nonlinear relationships and generally performs well on small to medium tabular datasets.

## 9. Implementation
The project contains a dataset generator, model trainer, prediction utility, Streamlit application, and serialized model files. Feature preprocessing is included inside the pipeline to keep training and prediction transformations consistent.

## 10. Results and Discussion
Actual MAE, R², accuracy, and confusion matrix values are generated from held-out test data and displayed in the application. Regression evaluates continuous marks; classification evaluates category labels.

## 11. Advantages
- Uses real trained models.
- Provides useful visual and rule-based guidance.
- Works locally without APIs or API keys.
- Beginner-readable implementation.

## 12. Limitations
- Synthetic data may not represent real institutions.
- The classifier category is derived from the final score threshold, so it is correlated with the regression target.
- Recommendations are general and rule-based, not personalized professional advice.
- The model cannot account for every social or psychological factor.

## 13. Future Scope
- Collect verified institution data.
- Add explainability for individual predictions.
- Add user authentication and database storage.
- Include student learning style or test-frequency features.
- Compare additional algorithms.

## 14. Conclusion
The project provides an understandable machine learning application for predicting student performance and identifying potential academic support needs.

## Viva Questions and Answers
**What is regression?** It predicts continuous numeric values, such as final marks.

**What is classification?** It assigns observations to categories, such as Excellent or Good.

**Why Random Forest?** It combines multiple decision trees and handles nonlinear relationships.

**What is train-test split?** It separates data so model performance is checked on records not used for training.

**What is MAE?** The average absolute difference between predicted and actual marks.

**What is R²?** It indicates how much variation in final marks is explained by the model.

**How is data leakage avoided?** The target columns are excluded from the input features and the test set remains separate during evaluation.
