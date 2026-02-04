Here is the complete `README.md` file content. You can copy the code block below directly into your GitHub file editor.

```markdown
# ChEMBL Kinase QSAR Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview

This repository houses an end-to-end Machine Learning pipeline designed to predict the lipophilicity (**LogP**) of kinase inhibitors targeting the **PI3K/AKT/mTOR signaling pathway**.

By leveraging the [ChEMBL Database](https://www.ebi.ac.uk/chembl/), this project automates the retrieval of bioactivity data, engineers molecular features from chemical formulas, and trains a Quantitative Structure-Activity Relationship (QSAR) model using optimized Decision Trees.

### 🎯 Biological Context
The PI3K/AKT/mTOR pathway is a critical regulator of cell cycle regulation and is frequently dysregulated in human cancers. Developing inhibitors with optimal physicochemical properties (like LogP) is essential for drug bioavailability.

---

## ⚙️ The Pipeline Architecture

The project is structured into three modular stages:

```mermaid
graph LR
    A[01_Data_Retrieval] -->|Raw CSV| B[02_Data_Cleaning]
    B -->|Cleaned Excel| C[03_Model_Training]
    C -->|Metrics & Plots| D[Final Model]

```

### 1. Data Retrieval (`dataretreival_chemebl_final_2.ipynb`)

* **Objective:** Mining bioactive molecules from the ChEMBL database.
* **Tech Stack:** `chembl_webresource_client`, `pandas`.
* **Key Actions:**
* Connects to ChEMBL API.
* Filters for human targets: **PI3K (  )**, **AKT**, and **mTOR**.
* Extracts canonical SMILES, Molecular Weight, and standard types (IC50, Ki).
* Aggregates data into a master raw dataset.



### 2. Feature Engineering (`2_DataCleaning.ipynb`)

* **Objective:** Transforming chemical strings into numerical feature vectors.
* **Tech Stack:** `pandas`, `re` (Regular Expressions).
* **Key Actions:**
* **Parsing:** Deconstructs `full_molformula` strings (e.g., ) into atomic counts.
* **Imputation:** Handles sparse data by imputing missing elemental counts with zero.
* **Formatting:** Exports a clean, densified dataset ready for ML ingestion.



### 3. Model Training & Optimization (`Model_training.ipynb`)

* **Objective:** Predicting LogP using supervised learning.
* **Tech Stack:** `scikit-learn`, `seaborn`, `matplotlib`.
* **Key Actions:**
* **Splitting:** 80/20 Train-Test split.
* **Algorithm:** Decision Tree Regressor.
* **Optimization:** 5-Fold Grid Search Cross-Validation (tuning `max_depth`, `min_samples_split`, `min_samples_leaf`).
* **Diagnostics:** Sensitivity analysis plotting RMSE vs. Complexity to prevent overfitting.



---

## 📊 Results

The model was evaluated on unseen test data after hyperparameter tuning.

| Metric | Score | Description |
| --- | --- | --- |
| **RMSE** | *[Insert Your RMSE]* | Average deviation between predicted and actual LogP. |
| **R² Score** | *[Insert Your R2]* | Variance explained by the model (max 1.0). |

*Visualizations of the hyperparameter stability landscape can be found in the `Model_training` notebook.*

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn chembl_webresource_client seaborn matplotlib

```

### Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/chembl-kinase-qsar-pipeline.git](https://github.com/your-username/chembl-kinase-qsar-pipeline.git)

```


2. **Run the notebooks in order:**
* Start with `dataretreival_chemebl_final_2.ipynb` to fetch fresh data.
* Run `2_DataCleaning.ipynb` to process the raw CSV.
* Run `Model_training.ipynb` to train and evaluate the model.



---

## 🔮 Future Improvements

* **Feature Expansion:** Integrating Morgan Fingerprints (ECFP4) for deeper structural analysis.
* **Model Comparison:** Benchmarking Decision Trees against Random Forests and XGBoost.
* **Deployment:** wrapping the model in a Streamlit app for real-time LogP prediction of user-inputted SMILES.

---

## 📝 Author

**Joshua Stein**

* [LinkedIn Profile]((https://www.linkedin.com/in/joshua-stein-3401b7129/))
* [Portfolio Link]((https://github.com/JoshuaBStein/machine_learning_portfolio))

```

```
