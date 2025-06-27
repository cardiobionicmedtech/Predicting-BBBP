# 🧠 Blood-Brain Barrier (BBB) Permeability Predictor

This project provides a **Streamlit-based web application** that predicts the permeability of molecules across the blood-brain barrier (BBB) using machine learning models and molecular descriptors. Users can upload CSV files with SMILES strings and receive permeability predictions, probability scores, visualizations, and structure-based analysis.

![BBB Predictor UI](https://github.com/user-attachments/assets/927a4aaf-e11d-4794-9b3d-9aeac78ac092)<!-- Add if available -->

---

## 🚀 Features

- ⚗️ Predict BBB permeability from SMILES strings.
- 🧬 Uses **molecular descriptors** and **Morgan fingerprints**.
- 🤖 Choose between:
  - `Extra Trees Classifier`
  - `Deep Neural Network (PyTorch)`
- 📊 Interactive dashboards (probability distribution, prediction ratios, boxplots).
- 📋 Individual molecular structure cards with confidence level.
- 📥 Downloadable results as CSV.
- 🖼️ Automatic rendering of molecular structures.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bbb-permeability-predictor.git
cd bbb-permeability-predictor
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
---

## 📁 Project Structure

```
📦 bbb-permeability-predictor/
├── frontend.py               # Main Streamlit app
├── app.ipynb                 # Notebook for development/test
├── best_bbbp_model_Extra_Trees.joblib
├── best_dnn_model.pth
├── final_dnn_model.pth
├── dnn_scaler.joblib
├── requirements.txt
```

---

## 🧪 How to Use

### Run Locally

```bash
streamlit run frontend.py
```

* Upload a CSV file with a column containing **SMILES strings**.
* Select a model (Extra Trees / DNN).
* Click **"Run Prediction"**.
* View predictions, statistics, graphs, and download the results.

### CSV Format

| compound\_id | smiles                        | compound\_name |
| ------------ | ----------------------------- | -------------- |
| COMP001      | CCO                           | Ethanol        |
| COMP002      | CC(C)CC1=CC=C(C=C1)C(C)C(=O)O | Ibuprofen      |

> ✅ Ensure at least one column contains "smiles" in its name.

---

## 🤖 Models

* **Extra Trees Classifier** (`best_bbbp_model_Extra_Trees.joblib`)
* **Deep Neural Network (PyTorch)** (`final_dnn_model.pth`, `dnn_scaler.joblib`)

All models are optionally loadable from the [HuggingFace Hub](https://huggingface.co/)

---

## 📊 Visualizations

* Donut chart showing permeable vs non-permeable predictions.
* Probability distribution histogram.
* Box plots for key descriptors: MW, LogP, TPSA, etc.
* Individual molecule detail cards with probability gauge.

---

## 🧠 Motivation

The blood-brain barrier is crucial in drug design. This tool aims to assist medicinal chemists and researchers in **screening molecular libraries** for CNS drug-likeness by estimating BBB permeability efficiently using ML.

---

## 🙌 Acknowledgements

* [RDKit](https://www.rdkit.org/)
* [Streamlit](https://streamlit.io/)
* [HuggingFace Hub](https://huggingface.co/)
* [PyTorch](https://pytorch.org/)
* [Plotly](https://plotly.com/python/)

```
