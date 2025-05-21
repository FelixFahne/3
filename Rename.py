import os
import argparse
import pandas as pd

def rename_and_record_files(input_dir: str, output_dir: str) -> None:
    """Rename Excel files sequentially and save a record Excel file."""

    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(".xlsx") or f.lower().endswith(".xls")
    ]

    records = []

    for i, file in enumerate(files, start=1):
        new_name = f"G{i}.xlsx"
        original_path = os.path.join(input_dir, file)
        new_path = os.path.join(output_dir, new_name)

        os.rename(original_path, new_path)
        records.append({"Original Name": file, "New Name": new_name})

    df = pd.DataFrame(records)
    df.to_excel(os.path.join(output_dir, "renaming_record.xlsx"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename Excel files sequentially")
    parser.add_argument(
        "--input-dir",
        default="Renamed_output",
        help="Directory containing the Excel files to rename",
    )
    parser.add_argument(
        "--output-dir",
        default="Renamed_output",
        help="Directory where renamed files and the record will be placed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rename_and_record_files(args.input_dir, args.output_dir)
