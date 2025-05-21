"""Utility script for converting Excel annotation files to CSV.

The original version of this script hard-coded absolute file paths.  It now
accepts an input directory containing Excel files and an output directory for
the converted CSV files.  Each Excel file found in the input directory will be
converted to a CSV file with the same base name in the output directory.

Example usage::

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
        csv_filename = os.path.splitext(excel_filename)[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Explicitly specify openpyxl as the engine for .xlsx files. Some of the
        # provided files may actually be legacy Excel formats. In that case,
        # fall back to the xlrd engine which handles .xls files.
        try:
            df = pd.read_excel(excel_path, engine="openpyxl")
        except Exception as e:
            print(f"  failed with openpyxl: {e}")
            try:
                df = pd.read_excel(excel_path, engine="xlrd")
            except Exception as e2:
                print(f"  skipped {excel_filename} due to read error: {e2}")
                continue
        df.to_csv(csv_path, index=False)
        csv_files.append(csv_path)

    print("All files have been converted to CSV format.")
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

    # Load one of the generated CSVs if available so subsequent analysis does not
    # fail. Replace or extend this step with your own data loading logic.
    if csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        df = pd.DataFrame()

# Placeholder DataFrame. Replace this with your own summarised data structure
# that aggregates label counts for each dialogue segment.
    summary_label_counts_by_segment = pd.DataFrame()


    def assign_tone(row):
        if row['backchannels'] > 0 or row['code-switching for communicative purposes'] > 0 or row['collaborative finishes'] > 0:
            return 'Informal'
        elif row['subordinate clauses'] > 0 or row['impersonal subject + non-factive verb + NP'] > 0:
            return 'Formal'
        else:
            return 'Neutral'  # Default to Neutral if none of the criteria match

# Apply the function to each segment
    summary_label_counts_by_segment['Tone'] = summary_label_counts_by_segment.apply(assign_tone, axis=1)

# Overview of tone assignments
    tone_assignments = summary_label_counts_by_segment['Tone'].value_counts()

# Visualize the distribution of assigned tones across segments
    plt.figure(figsize=(8, 5))
    tone_assignments.plot(kind='bar')
    plt.title('Distribution of Assigned Tones Across Dialogue Segments')
    plt.xlabel('Tone')
    plt.ylabel('Number of Segments')
    plt.xticks(rotation=0)
    plt.show()

# Now that we have assigned tones, we could explore the relationship between these tones and specific labels
# This step is illustrative and based on the simplified criteria for tone assignment
    tone_assignments


# Assuming `df` is a DataFrame with dialogue identifiers and the constructed dialogue-level labels

# Feature Engineering: Summarize token-level labels into dialogue-level features
    features = df.groupby('dialogue_id').agg({
    'token_label_type1': 'sum',
    'token_label_type2': 'sum',
    # Add more as needed
})

# Assume `dialogue_labels` is a DataFrame with our dialogue-level labels
    dialogue_labels = df.groupby('dialogue_id').agg({
    'OverallToneChoice': 'first',  # Assuming a method to assign these labels
    'TopicExtension': 'first'
})

# Join features with labels
    data_for_regression = features.join(dialogue_labels)

# Split data into features (X) and labels (y)
    X = data_for_regression.drop(['OverallToneChoice', 'TopicExtension'], axis=1)
    y = data_for_regression[['OverallToneChoice', 'TopicExtension']]

# Regression analysis (simplified)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

# Splitting dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model for 'Overall Tone Choice'
    model_tone = LinearRegression().fit(X_train, y_train['OverallToneChoice'])

# Predict and evaluate 'Overall Tone Choice'
# (Evaluation steps would go here)

# Repeat for 'Topic Extension'
    model_topic = LinearRegression().fit(X_train, y_train['TopicExtension'])

    # Predict and evaluate 'Topic Extension'


if __name__ == "__main__":
    main()

