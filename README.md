# TESI-DENTIPOMPIANI
Analisi di cluster e modelli predittivi per l’identificazione di biomarker in ambito biomedico

---

## 🧠 Overview of Scripts

### `corr-analisi-prova.py`
Performs **correlation analysis** and **feature selection**:
- Computes the correlation matrix.  
- Removes features with |r| > 0.8 to reduce collinearity.  
- Produces a heatmap and a list of selected independent variables.

### `multimodello.py`
Compares four supervised learning algorithms:
- **Random Forest**  
- **Logistic Regression**  
- **Support Vector Machine (SVM)**  
- **XGBoost**

Each model is evaluated through:
- **Accuracy**, **F1-score**, and **AUPRC (Area Under Precision-Recall Curve)**.  
The results are summarized in **barcharts** comparing model performance for males, females, and overall means.

### `final-shap.py`
Implements the **complete analysis pipeline**, including model interpretation:
- Training of **Random Forest** and **Logistic Regression** with repeated cross-validation (11 iterations).  
- Calculation of **Permutation Importance** and **SHAP values**.  
- Automatic generation of:
  - Comparative forest plots (RF vs LR)
  - Normalized SHAP vs Permutation plots
  - Individual **SHAP waterfall plots**
- Optional flag `USE_TEST_FOR_PERM` allows computation on Train or Test sets.

---

## ⚙️ Requirements

Python ≥ 3.9  
Install dependencies via:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap xgboost openpyxl

