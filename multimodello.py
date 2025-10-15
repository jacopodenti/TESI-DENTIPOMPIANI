import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score,
    precision_score, average_precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# Config

FILE_PATH = "/Users/gabriele/Desktop/TESI/CMS-databehaviour_males-females 1.xlsx"
SHEETS = ["maschi", "femmine"]   
USECOLS = "C:Q"                  # intervallo colonne come nel file
HEADER_ROW = 2                   
NO_ITERS = 11
SKF = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

FEATURES = [
    'body weight', 'sucrose intake', 'NOR index', 'locomotor activity',
    'social interaction time', 'social events', '0P (entries)',
    'CL (entries)', 'tCL', 'tCENT'
]

MODELS = {
    "RF": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    "LR": LogisticRegression(
        penalty='l2', solver='liblinear', class_weight='balanced', random_state=42
    ),
    "SVM": SVC(
        kernel='rbf', probability=True, class_weight='balanced', random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.0, random_state=42
    ),
}


# Helpers

def evaluate_models_on_df(df: pd.DataFrame):
    df = df.dropna().copy()
    df['stress'] = df['stress'].astype(int)
    X = df[FEATURES]
    y = df['stress']

    metrics_summary = {}
    avg_confusions = {}

    for model_name, clf in MODELS.items():
        performance_metrics = []
        conf_matrices = []

        for _ in range(NO_ITERS):
            y_pred_labels = np.empty(len(X), dtype=int)
            y_pred_probas = np.empty(len(X))

            for train_idx, test_idx in SKF.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                clf.fit(X_train, y_train)
                y_pred_labels_fold = clf.predict(X_test)
                y_pred_proba_fold = clf.predict_proba(X_test)[:, 1]

                y_pred_labels[test_idx] = y_pred_labels_fold
                y_pred_probas[test_idx] = y_pred_proba_fold

            acc = accuracy_score(y, y_pred_labels)
            f1 = f1_score(y, y_pred_labels)
            recall = recall_score(y, y_pred_labels)
            precision = precision_score(y, y_pred_labels)
            roc_auc = roc_auc_score(y, y_pred_probas)
            auprc = average_precision_score(y, y_pred_probas)
            cm = confusion_matrix(y, y_pred_labels)

            performance_metrics.append({
                'accuracy': acc, 'f1': f1, 'recall': recall,
                'precision': precision, 'roc_auc': roc_auc, 'auprc': auprc
            })
            conf_matrices.append(cm)

        metrics_mean = {k: float(np.mean([m[k] for m in performance_metrics]))
                        for k in performance_metrics[0].keys()}
        avg_cm = np.round(np.mean(conf_matrices, axis=0)).astype(int)

        metrics_summary[model_name] = metrics_mean
        avg_confusions[model_name] = avg_cm

    return metrics_summary, avg_confusions, len(df)


def plot_barchart(metrics_summary, sex_label, filename):
    models_order = ["RF", "LR", "SVM", "XGBoost"]
    auprc_vals = [metrics_summary[m]["auprc"] for m in models_order]
    f1_vals    = [metrics_summary[m]["f1"]    for m in models_order]

    x = np.arange(len(models_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, auprc_vals, width, label="AUPRC")
    ax.bar(x + width/2, f1_vals,    width, label="F1-score")

    ax.set_ylim(0, 1.10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_order)
    ax.set_ylabel("Score")
    ax.set_title(f"Confronto modelli ({sex_label})")
    ax.legend(loc="upper left")

    for i, v in enumerate(auprc_vals):
        ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(f1_vals):
        ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_barchart_aggregate(metrics_maschi, n_maschi, metrics_femmine, n_femmine, filename):
    """NUOVO BLOCCO MEDIA: media pesata per numerosità campione (se uguali, coincide con la media semplice)."""
    models_order = ["RF", "LR", "SVM", "XGBoost"]
    w1, w2 = n_maschi, n_femmine
    wtot = w1 + w2

    auprc_vals = []
    f1_vals = []
    for m in models_order:
        auprc_avg = (metrics_maschi[m]["auprc"] * w1 + metrics_femmine[m]["auprc"] * w2) / wtot
        f1_avg    = (metrics_maschi[m]["f1"]    * w1 + metrics_femmine[m]["f1"]    * w2) / wtot
        auprc_vals.append(auprc_avg)
        f1_vals.append(f1_avg)

    x = np.arange(len(models_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, auprc_vals, width, label="AUPRC (media)")
    ax.bar(x + width/2, f1_vals,    width, label="F1 (media)")

    ax.set_ylim(0, 1.10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_order)
    ax.set_ylabel("Score")
    ax.set_title("Confronto modelli – media complessiva (maschi+femmine)")
    ax.legend(loc="upper left")

    for i, v in enumerate(auprc_vals):
        ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(f1_vals):
        ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# ==========
# Main
# ==========
if __name__ == "__main__":
    results_for_sheet = {}

    # 1) Esecuzione separata per MASCHI e FEMMINE + barchart singoli
    for sheet in SHEETS:
        print(f"\n======== ANALISI GRUPPO: {sheet.upper()} ========")
        df = pd.read_excel(FILE_PATH, sheet_name=sheet, header=HEADER_ROW, usecols=USECOLS)

        metrics_summary, avg_confusions, nrows = evaluate_models_on_df(df)
        results_for_sheet[sheet] = (metrics_summary, avg_confusions, nrows)

        for model in ["RF", "LR", "SVM", "XGBoost"]:
            m = metrics_summary[model]
            cm = avg_confusions[model]
            print(f"\n--- {model} ---")
            print(f"Accuracy:  {m['accuracy']:.4f}")
            print(f"F1-Score:  {m['f1']:.4f}")
            print(f"Recall:    {m['recall']:.4f}")
            print(f"Precision: {m['precision']:.4f}")
            print(f"ROC AUC:   {m['roc_auc']:.4f}")
            print(f"AUPRC:     {m['auprc']:.4f}")
            print(f"Confusion Matrix media:\n{cm}")

        label = "Maschi" if sheet.lower().startswith("maschi") else "Femmine"
        filename = f"barchart_multimodello_{label.lower()}.png"
        plot_barchart(metrics_summary, label, filename)
        print(f"[BARCHART] Salvato: {filename}")

    # 2) Barchart media complessiva (maschi + femmine)
    m_metrics, _, n_m = results_for_sheet["maschi"]
    f_metrics, _, n_f = results_for_sheet["femmine"]
    plot_barchart_aggregate(m_metrics, n_m, f_metrics, n_f, "barchart_multimodello_media.png")
    print("[BARCHART] Salvato: barchart_multimodello_media.png")
