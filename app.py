import tempfile
import joblib
import gradio as gr
from gradio.themes.base import Base
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the annotation tool HTML and inline the JavaScript so it works inside Gradio
with open("Annotation_tool.html", "r", encoding="utf-8") as f:
    annotation_html = f.read()

with open("script.js", "r", encoding="utf-8") as f:
    script_js = f.read()

annotation_html = annotation_html.replace(
    '<script src="script.js"></script>', f"<script>{script_js}</script>"
)


def preprocess_excel(file):
    """Convert an uploaded Excel or CSV file to CSV and return the path."""
    if file is None:
        raise gr.Error("Please upload an Excel or CSV file")

    filename = file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name, engine="openpyxl")
    except Exception as e:
        raise gr.Error(f"Failed to read file: {e}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name


def train_model(csv_file):
    """Train a simple model on the provided CSV."""
    if csv_file is None:
        raise gr.Error("Please upload a CSV file")
    df = pd.read_csv(csv_file.name)
    if "label" not in df.columns:
        raise gr.Error("CSV must contain a column named 'label'")
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl").name
    joblib.dump(clf, model_path)
    return acc, model_path


def evaluate_model(model_file, csv_file):
    """Evaluate a trained model on a CSV file."""
    if model_file is None or csv_file is None:
        raise gr.Error("Please provide both a model file and a CSV file")
    clf = joblib.load(model_file.name)
    df = pd.read_csv(csv_file.name)
    if "label" not in df.columns:
        raise gr.Error("CSV must contain a column named 'label'")
    X = df.drop(columns=["label"])
    y = df["label"]
    preds = clf.predict(X)
    acc = float(accuracy_score(y, preds))
    return acc


demo = gr.Blocks(theme=Base())

with demo:
    gr.Markdown("# SLEDA Tools")
    with gr.Tab("Annotation"):
        gr.HTML(annotation_html)

    with gr.Tab("Preprocessing"):
        input_excel = gr.File(label="Excel or CSV file")
        convert_btn = gr.Button("Convert to CSV")
        output_csv = gr.File(label="CSV output")
        convert_btn.click(preprocess_excel, inputs=input_excel, outputs=output_csv)

    with gr.Tab("Training"):
        train_csv = gr.File(label="Training CSV (must include 'label' column)")
        train_btn = gr.Button("Train model")
        train_accuracy = gr.Number(label="Accuracy")
        model_output = gr.File(label="Model file")
        train_btn.click(train_model, inputs=train_csv, outputs=[train_accuracy, model_output])

    with gr.Tab("Evaluation"):
        model_file = gr.File(label="Model file (.pkl)")
        eval_csv = gr.File(label="Evaluation CSV (must include 'label' column)")
        eval_btn = gr.Button("Evaluate")
        eval_accuracy = gr.Number(label="Accuracy")
        eval_btn.click(evaluate_model, inputs=[model_file, eval_csv], outputs=eval_accuracy)

if __name__ == "__main__":
    demo.launch()
