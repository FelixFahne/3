import tempfile
import joblib
import gradio as gr
from gradio.themes.base import Base
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────
def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lower-case column names for consistent access."""
    df.rename(columns=lambda c: c.strip().lower(), inplace=True)
    return df


# ──────────────────────────────────────────────────────────────
# Tab 1 · Preprocessing  (Excel/CSV  →  numerische CSV)
# ──────────────────────────────────────────────────────────────
def preprocess_excel(file):
    """
    Upload Excel/CSV → gibt eine numerische CSV zurück.

    Kategoriale Spalten werden automatisch One-Hot-kodiert,
    die 'label'-Spalte bleibt unverändert.
    """
    if file is None:
        raise gr.Error("Please upload an Excel or CSV file")

    # 1) Laden --------------------------------------------------
    filename = file.name.lower()
    try:
        df = pd.read_csv(file.name) if filename.endswith(".csv") else pd.read_excel(
            file.name, engine="openpyxl"
        )
    except Exception as e:
        raise gr.Error(f"Failed to read file: {e}")

    canonicalize_columns(df)

    # 2) One-Hot-Encoding (label ausnehmen) ---------------------
    label_col = "label" if "label" in df.columns else None
    cat_cols = (
        df.select_dtypes(include=["object", "category"])
        .columns.difference([label_col])
    )
    if len(cat_cols):
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # -----------------------------------------------------------

    # 3) Temp-CSV zurückgeben
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name


# ──────────────────────────────────────────────────────────────
# Tab 2 · Training
# ──────────────────────────────────────────────────────────────
def train_model(csv_file):
    """Train a RandomForest on the provided (numerischen) CSV."""
    if csv_file is None:
        raise gr.Error("Please upload a CSV or Excel file")

    # read file -------------------------------------------------
    filename = csv_file.name.lower()
    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_file.name, engine="openpyxl")
    else:
        df = pd.read_csv(csv_file.name, encoding="utf-8-sig")

    canonicalize_columns(df)

    # sanity checks --------------------------------------------
    if "label" not in df.columns:
        raise gr.Error("CSV must contain a column named 'label'")

    X = df.drop(columns=["label"])
    bad = X.select_dtypes(include=["object", "category"]).columns
    if len(bad):
        raise gr.Error(
            f"Spalten nicht numerisch: {list(bad)}. "
            "Bitte die Datei aus dem Preprocessing-Tab benutzen."
        )

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


# ──────────────────────────────────────────────────────────────
# Tab 3 · Evaluation
# ──────────────────────────────────────────────────────────────
def evaluate_model(model_file, csv_file):
    """Evaluate a trained model on a numerische CSV Datei."""
    if model_file is None or csv_file is None:
        raise gr.Error("Please provide both a model file and a CSV file")

    clf = joblib.load(model_file.name)

    # read file -------------------------------------------------
    filename = csv_file.name.lower()
    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_file.name, engine="openpyxl")
    else:
        df = pd.read_csv(csv_file.name, encoding="utf-8-sig")

    canonicalize_columns(df)

    # sanity checks --------------------------------------------
    if "label" not in df.columns:
        raise gr.Error("CSV must contain a column named 'label'")

    X = df.drop(columns=["label"])
    bad = X.select_dtypes(include=["object", "category"]).columns
    if len(bad):
        raise gr.Error(
            f"Spalten nicht numerisch: {list(bad)}. "
            "Bitte die Datei aus dem Preprocessing-Tab benutzen."
        )

    y = df["label"]
    preds = clf.predict(X)
    acc = float(accuracy_score(y, preds))
    return acc


# ──────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────
demo = gr.Blocks(theme=Base())

with demo:
    gr.Markdown("# SLEDA Tools")

    # Tab 1 · Preprocessing
    with gr.Tab("Preprocessing"):
        input_excel = gr.File(label="Excel or CSV file")
        convert_btn = gr.Button("Convert to numeric CSV")
        output_csv = gr.File(label="Numeric CSV output")
        convert_btn.click(preprocess_excel, inputs=input_excel, outputs=output_csv)

    # Tab 2 · Training
    with gr.Tab("Training"):
        train_csv = gr.File(
            label="Training CSV (must include 'label' column; use output from Tab 1)"
        )
        train_btn = gr.Button("Train model")
        train_accuracy = gr.Number(label="Accuracy")
        model_output = gr.File(label="Model file (.pkl)")
        train_btn.click(
            train_model,
            inputs=train_csv,
            outputs=[train_accuracy, model_output],
        )

    # Tab 3 · Evaluation
    with gr.Tab("Evaluation"):
        model_file = gr.File(label="Model file (.pkl)")
        eval_csv = gr.File(
            label="Evaluation CSV (must include 'label' column; use output from Tab 1)"
        )
        eval_btn = gr.Button("Evaluate")
        eval_accuracy = gr.Number(label="Accuracy")
        eval_btn.click(
            evaluate_model,
            inputs=[model_file, eval_csv],
            outputs=eval_accuracy,
        )

if __name__ == "__main__":
    demo.launch()
