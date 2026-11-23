# Smart Student Performance Predictor

A beginner-friendly Streamlit machine learning project that predicts final student marks and performance categories.

## Project Structure

```text
smart_student_predictor/
├── app.py
├── train_model.py
├── generate_dataset.py
├── student_performance_dataset.csv
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── models/
│   ├── performance_regressor.pkl
│   ├── performance_classifier.pkl
│   └── model_metrics.json
└── utils/
    └── prediction.py
```

## Windows Installation

1. Install Python 3.10 or newer from https://www.python.org/downloads/. Select **Add Python to PATH**.
2. Open the project folder in VS Code or File Explorer.
3. Open a terminal and run:

```powershell
cd "c:\Users\HP\OneDrive\Desktop\JAVA\smart_student_predictor"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Generate and Train

```powershell
python generate_dataset.py
python train_model.py
```

## Run the Application

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Sample Input

- Attendance: 72%
- Study hours: 2.5
- Internal marks: 58
- Assignment score: 62
- Previous semester marks: 64

## Troubleshooting

- **Module not found:** Activate `.venv`, then run `pip install -r requirements.txt`.
- **Execution policy blocked:** Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- **Missing models:** Run `python train_model.py`.
- **Missing dataset:** Run `python generate_dataset.py`.
- **Streamlit not recognized:** Run `python -m streamlit run app.py`.

## Regression vs Classification

Regression predicts a numeric final mark. Classification predicts one of four named performance categories.
