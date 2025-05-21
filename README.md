---
title: SLEDA Tools
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.30.0"
app_file: app.py
pinned: false
---
# 2024InteractiveMetrics

This repository contains code and sample data for experimenting with the SLEDA framework. SLEDA is a three level approach for analysing English Second Language (ESL) dialogues.

The repository currently provides a lightweight setup with a few core tools and example notebooks. It does not include the larger directory layout described in older versions of the README.

## Contents

- **`SLDEA Data/`** – Excel files with dialogue annotations.
- **`preprocessing.py`** – converts the Excel files to CSV format.
- **`Annotation_tool.html` / `script.js`** – a small web app for manual annotation.
- **`libs/xlsx.full.min.js`** – local copy of the SheetJS library required by the annotation tool.
- **`dialogue_pred.ipynb`** – example notebook for training a model.
- **`ESL_AddedExperinments.ipynb`** – additional experiments.
- **`2024ACLESLMainCodes_Results/`** – CSV files with example results.
- **`feature_label.csv`** – features extracted from the annotated data.

## Installation

Install the required packages with pip:

```bash
pip install -r requirements.txt
```

## Basic Workflow

1. **Convert the Excel data**
   ```bash
   python preprocessing.py --input-dir "SLDEA Data" --output-dir csv_output
   ```
   This will create a `csv_output` folder with the converted CSV files. Pass
   `--convert-only` if you want to skip the analysis section of the script.

2. **Annotate**
   Open `Annotation_tool.html` in your browser. You can load CSV files and apply labels using the interface.
   The page relies on the SheetJS library provided in `libs/xlsx.full.min.js` to
   read Excel files. If the library is missing, download it from
   `https://cdn.jsdelivr.net/npm/xlsx/dist/xlsx.full.min.js` and place it in the
   `libs/` folder.

3. **Train a model**
   Use `dialogue_pred.ipynb` (or `ESL_AddedExperinments.ipynb`) in Jupyter Notebook. Load the CSV files created in step 1 and follow the cells to train a simple model.

4. **Evaluate**
   After training, you can run evaluation steps in the notebook or compare results with the CSVs inside `2024ACLESLMainCodes_Results`.

This simplified workflow should help you get started even if you are new to Python. The larger directory structure mentioned in earlier READMEs (e.g. `dataset/SLDEA`, `notebooks/`, `figures/`, `utils/`, `reports/`) is not present in this repository.

## Running in a Hugging Face Space

You can launch the included `app.py` script on Hugging Face Spaces or locally. It provides a small Gradio interface with four tabs:

1. **Annotation** – shows the existing annotation tool directly in the browser.
2. **Preprocessing** – upload an Excel **or** CSV file and download the converted CSV.
3. **Training** – upload a CSV (with a `label` column) to train a simple model. The accuracy and a downloadable model file are returned.
4. **Evaluation** – upload a trained model file together with a CSV to measure accuracy on that data.

To run the app locally, install the dependencies and execute:

```bash
python app.py
```

The interface will open in your browser. On Hugging Face Spaces you can add all repository files and set `app.py` as the entry point.

## Step-by-Step Guide

If you are new to Python, follow these steps to run each tool in order:

1. **Install Python** – Download and install Python 3.8 or newer from [python.org](https://www.python.org/).
2. **Install dependencies** – Open a terminal in this repository and run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess the data** – Convert the Excel files by executing:
   ```bash
   python preprocessing.py --input-dir "SLDEA Data" --output-dir csv_output
   ```
   The converted CSV files will appear inside the `csv_output` folder. Use
   the optional `--convert-only` flag to run just the conversion step.
4. **Annotate dialogues** – Double click `Annotation_tool.html`. Your browser will open the annotation interface. Load a CSV file, apply labels, and save your work.
5. **Train a model** – Open either `dialogue_pred.ipynb` or `ESL_AddedExperinments.ipynb` in Jupyter Notebook and run the cells one by one using the annotated CSV files from the previous step. A model file will be generated.
6. **Evaluate a corpus** – Use the evaluation tab in `app.py` or the final notebook cells to load the saved model together with new CSV data. Accuracy numbers will be printed.
