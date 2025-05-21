"""Utility script for converting Excel annotation files to *numeric* CSV.

The original version hard-coded absolute paths.  
Jetzt werden alle .xlsx/.xls aus einem Eingabeordner eingelesen und zwei
Dateien erzeugt:

1. <name>.csv       – 1-zu-1-Dump des Excel (zur Referenz)
2. <name>_num.csv   – numerische Variante mit One-Hot-Encoding für
                      alle Objekt-/Kategorie-Spalten **außer `label`**

Beispiel ::

    python preprocessing.py --input-dir "SLDEA Data" --output-dir csv_output
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def convert_excels_to_csv(input_dir: str, output_dir: str):
    """Convert all Excel files in *input_dir* to CSV files in *output_dir*."""

    os.makedirs(output_dir, exist_ok=True)

    excel_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith((".xlsx", ".xls"))
    ]

    csv_files = []
    for excel_filename in excel_files:
        excel_path = os.path.join(input_dir, excel_filename)
        print(f"Processing {excel_path}")

        # ------------------------------------------------------------
        # Schritt 1: Excel einlesen (openpyxl → xlrd Fallback)
        # ------------------------------------------------------------
        try:
            df = pd.read_excel(excel_path, engine="openpyxl")
        except Exception as e:
            print(f"  failed with openpyxl: {e}")
            try:
                df = pd.read_excel(excel_path, engine="xlrd")
            except Exception as e2:
                print(f"  skipped {excel_filename} due to read error: {e2}")
                continue

        # ------------------------------------------------------------
        # Schritt 2: Raw-CSV speichern (optional, Referenz)
        # ------------------------------------------------------------
        csv_filename = os.path.splitext(excel_filename)[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)

        # ------------------------------------------------------------
        # Schritt 3: Objekt-/Kategorie-Spalten One-Hot-Encodieren
        #            (label bleibt unverändert)
        # ------------------------------------------------------------
        label_col = "label" if "label" in df.columns else None
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if label_col in cat_cols:
            cat_cols = cat_cols.drop(label_col)

        if len(cat_cols):
            df_num = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        else:
            df_num = df.copy()

        num_filename = os.path.splitext(excel_filename)[0] + "_num.csv"
        num_path = os.path.join(output_dir, num_filename)
        df_num.to_csv(num_path, index=False)

        print(f"  ✓ wrote {num_path}")
        csv_files.append(num_path)

    print("All files have been converted to *_num.csv format.")
    return csv_files


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Excel files to CSV")
    parser.add_argument(
        "--input-dir",
        default="SLDEA Data",
        help="Directory containing the Excel annotation files",
    )
    parser.add_argument(
        "--output-dir",
        default="csv_output",
        help="Directory where converted CSV files will be placed",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert Excel files and skip the analysis step",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_files = convert_excels_to_csv(args.input_dir, args.output_dir)

    if args.convert_only:
        return

    # Load one of the generated numeric CSVs for follow-up analysis
    if csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        df = pd.DataFrame()

    # ----------------------------------------------------------------
    # ----- Beispiel-Analyse (bestehender Code bleibt unverändert) ----
    # ----------------------------------------------------------------
    summary_label_counts_by_segment = pd.DataFrame()

    def assign_tone(row):
        if (
            row["backchannels"] > 0
            or row["code-switching for communicative purposes"] > 0
            or row["collaborative finishes"] > 0
        ):
            return "Informal"
        elif (
            row["subordinate clauses"] > 0
            or row["impersonal subject + non-factive verb + NP"] > 0
        ):
            return "Formal"
        else:
            return "Neutral"

    summary_label_counts_by_segment["Tone"] = summary_label_counts_by_segment.apply(
        assign_tone, axis=1
    )

    tone_assignments = summary_label_counts_by_segment["Tone"].value_counts()

    plt.figure(figsize=(8, 5))
    tone_assignments.plot(kind="bar")
    plt.title("Distribution of Assigned Tones Across Dialogue Segments")
    plt.xlabel("Tone")
    plt.ylabel("Number of Segments")
    plt.xticks(rotation=0)
    plt.show()

    # ----------------------------------------------------------------
    # Example feature engineering + regression (unchanged)
    # ----------------------------------------------------------------
    features = df.groupby("dialogue_id").agg(
        {
            "token_label_type1": "sum",
            "token_label_type2": "sum",
            # Add more as needed
        }
    )

    dialogue_labels = df.groupby("dialogue_id").agg(
        {
            "OverallToneChoice": "first",
            "TopicExtension": "first",
        }
    )

    data_for_regression = features.join(dialogue_labels)

    X = data_for_regression.drop(["OverallToneChoice", "TopicExtension"], axis=1)
    y = data_for_regression[["OverallToneChoice", "TopicExtension"]]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_tone = LinearRegression().fit(X_train, y_train["OverallToneChoice"])
    model_topic = LinearRegression().fit(X_train, y_train["TopicExtension"])
    # (Evaluation code would go here)


if __name__ == "__main__":
    main()
