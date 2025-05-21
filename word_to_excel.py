import os
import re
import argparse
from docx import Document
import pandas as pd

def process_document_with_id(doc_path):
    # Load the Word document
    doc = Document(doc_path)

    # Combine paragraphs into a single text, removing all newline characters
    combined_text = ' '.join([p.text.replace('\n', ' ') for p in doc.paragraphs if p.text.strip() != ''])

    # Pattern to identify speakers and split the text accordingly
    pattern = r'(SPK_\d+|Rena)\s*'
    segments = re.split(pattern, combined_text)
    segments = [seg.strip() for seg in segments if seg.strip()]  # Removing any empty strings

    # Pairing speakers with their corresponding dialogue and adding dialogue ID
    data = []
    speaker = None
    dialogue_id_base = os.path.splitext(os.path.basename(doc_path))[0]  # Base for dialogue ID
    dialogue_count = 1  # Starting dialogue count

    for seg in segments:
        if seg in {'SPK_2', 'SPK_1', 'Rena'}:
            speaker = seg  # Update the speaker
        elif speaker:
            dialogue_id = f"{dialogue_id_base}_{dialogue_count}"
            data.append({'Dialogue ID': dialogue_id, 'Speaker': speaker, 'Content': seg})
            speaker = None  # Reset the speaker for the next dialogue
            dialogue_count += 1  # Increment dialogue count

    # Convert the data to a DataFrame
    return pd.DataFrame(data)

def process_all_documents(input_folder: str, output_folder: str) -> None:
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each Word document in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.docx'):
            doc_path = os.path.join(input_folder, filename)
            df = process_document_with_id(doc_path)

            # Save the DataFrame to an Excel file in the output folder
            output_file = os.path.join(output_folder, filename.replace('.docx', '.xlsx'))
            df.to_excel(output_file, index=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Word transcripts to Excel with dialogue IDs"
    )
    parser.add_argument(
        "--input-dir",
        default="Input",
        help="Directory containing the Word files",
    )
    parser.add_argument(
        "--output-dir",
        default="Output",
        help="Directory where the Excel files will be saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_documents(args.input_dir, args.output_dir)
