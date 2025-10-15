import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    recall_score, precision_score, average_precision_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone  # <-- per il fit full-data dei waterfall
import warnings, os
import shap

warnings.filterwarnings("ignore")

# === Percorso file ===
file_path = "/Users/gabriele/Desktop/TESI/CMS-databehaviour_males-females 1.xlsx"
output_dir = "/Users/gabriele/Desktop/TESI/tesi"
os.makedirs(output_dir, exist_ok=True)

# === Modelli ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
}

# === Feature base (default identiche per entrambi) ===
features = [
    'body weight', 'sucrose intake', 'NOR index', 'locomotor activity',
    'social interaction time', 'social events', '0P (entries)',
    'CL (entries)', 'tCL', 'tCENT'
]

# === Selezione manuale feature per gruppo (opzionale) ===
USE_ONLY = {
    'maschi': None,
    'femmine': None,
}
EXCLUDE = {
    'maschi': [],
    'femmine': [],
}

# === Metriche ===
metrics = ['f1', 'precision', 'recall', 'auprc']
SCORING_MAP = {
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'auprc': 'average_precision'
}

# === Flag: permutation sul TEST fold (True) oppure TRAIN fold (False)
USE_TEST_FOR_PERM = False
# === SHAP calcolati su TEST (True) o TRAIN (False)
USE_TEST_FOR_SHAP = False

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
no_iters = 11

# === Helpers SHAP & Forest plot ===
def shap_mean_abs_binary(clf, X_train, X_eval):
    """
    Mean(|SHAP|) per feature su X_eval.
    Se sv.values è (n,p,2) prende il canale della classe positiva.
    """
    try:
        explainer = shap.Explainer(clf, X_train)
        sv = explainer(X_eval)
        vals = sv.values
        if vals.ndim == 3:  # binario
            vals = vals[:, :, 1]
        return np.mean(np.abs(vals), axis=0)
    except Exception as e:
        print(f"[SHAP] Fallback generico: {e}")
        try:
            f = lambda data: clf.predict_proba(pd.DataFrame(data, columns=X_train.columns))[:, 1]
            kexp = shap.KernelExplainer(f, shap.sample(X_train, min(50, len(X_train)), random_state=0))
            sv = kexp.shap_values(X_eval, nsamples=100)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            return np.mean(np.abs(np.array(sv)), axis=0)
        except Exception as e2:
            print(f"[SHAP] KernelExplainer fallito: {e2}")
            return np.zeros(X_train.shape[1])

def forest_plot(features_list, perm_mean, perm_std, shap_mean, shap_std, title, outpath):
    y_pos = np.arange(len(features_list)); h = 0.18
    plt.figure(figsize=(9, max(4, 0.45*len(features_list))))
    # Permutation (AUPRC)
    plt.errorbar(perm_mean, y_pos + h, xerr=perm_std, fmt='o', ecolor='gray',
                 capsize=3, label='Permutation (AUPRC)')
    # SHAP
    plt.errorbar(shap_mean, y_pos - h, xerr=shap_std, fmt='o', ecolor='tab:red',
                 capsize=3, label='SHAP |mean|')
    plt.yticks(y_pos, features_list)
    plt.axvline(0, color='k', lw=0.8, ls='--')
    plt.xlabel("Importanza (media ± dev.std)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# === Forest plot singoli & normalizzati ===
def forest_plot_single(features_list, mean_vals, std_vals, title, outpath, label):
    y = np.arange(len(features_list))
    plt.figure(figsize=(9, max(4, 0.45*len(features_list))))
    plt.errorbar(mean_vals, y, xerr=std_vals, fmt='o', capsize=3, ecolor='gray', label=label)
    plt.yticks(y, features_list)
    plt.axvline(0, color='k', lw=0.8, ls='--')
    plt.xlabel("Importanza (media ± dev.std)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def forest_plot_single_normalized(features_list, mean_vals, std_vals, title, outpath, label):
    eps = 1e-12
    scale = max(np.max(np.abs(mean_vals)), eps)
    m = mean_vals / scale
    s = std_vals / scale
    y = np.arange(len(features_list))
    plt.figure(figsize=(9, max(4, 0.45*len(features_list))))
    plt.errorbar(m, y, xerr=s, fmt='o', capsize=3, ecolor='gray', label=f"{label} (norm)")
    plt.yticks(y, features_list)
    plt.axvline(0, color='k', lw=0.8, ls='--')
    plt.xlabel("Importanza normalizzata [0–1]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def forest_plot_normalized(features_list, perm_mean, perm_std, shap_mean, shap_std, title, outpath):
    eps = 1e-12
    p_scale = max(np.max(np.abs(perm_mean)), eps)
    s_scale = max(np.max(np.abs(shap_mean)), eps)
    perm_m = perm_mean / p_scale
    perm_s = perm_std  / p_scale
    shap_m = shap_mean / s_scale
    shap_s = shap_std  / s_scale
    y = np.arange(len(features_list)); h = 0.18
    plt.figure(figsize=(9, max(4, 0.45*len(features_list))))
    plt.errorbar(perm_m, y + h, xerr=perm_s, fmt='o', ecolor='gray', capsize=3, label='Permutation (norm)')
    plt.errorbar(shap_m, y - h, xerr=shap_s, fmt='o', ecolor='tab:red', capsize=3, label='SHAP |mean| (norm)')
    plt.yticks(y, features_list)
    plt.axvline(0, color='k', lw=0.8, ls='--')
    plt.xlabel("Importanza normalizzata [0–1]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# === Waterfall su 2 individui correttamente predetti (uno 0, uno 1) ===
def save_two_waterfalls(clf, X, y, features_list, sheet, model_name, outdir):
    """
    Fit su TUTTO il dataset del gruppo (per avere predizioni sullo stesso modello),
    scegli 1 negativo e 1 positivo corretti e salva due waterfall SHAP.
    """
    model_tag = model_name.replace(" ", "_")
    subdir = os.path.join(outdir, "waterfalls")
    os.makedirs(subdir, exist_ok=True)

    clf_full = clone(clf)
    clf_full.fit(X, y)
    y_hat = clf_full.predict(X)

    # trova indici corretti per classi 0 e 1
    idx_neg = next((i for i in range(len(y)) if y.iloc[i] == 0 and y_hat[i] == 0), None)
    idx_pos = next((i for i in range(len(y)) if y.iloc[i] == 1 and y_hat[i] == 1), None)
    if idx_neg is None or idx_pos is None:
        print("[WATERFALL] Non trovati esempi corretti per entrambe le classi, salto.")
        return

    # SHAP Explanation su due righe selezionate
    try:
        explainer = shap.Explainer(clf_full, X)
        sv = explainer(X.iloc[[idx_neg, idx_pos]])
    except Exception as e:
        print(f"[WATERFALL] Explainer rapido fallito: {e}")
        f = lambda data: clf_full.predict_proba(pd.DataFrame(data, columns=X.columns))[:, 1]
        kexp = shap.KernelExplainer(f, shap.sample(X, min(50, len(X)), random_state=0))
        vals = kexp.shap_values(X.iloc[[idx_neg, idx_pos]], nsamples=200)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        
        base_val = kexp.expected_value if not isinstance(kexp.expected_value, (list, tuple)) else kexp.expected_value[1]
        sv = shap.Explanation(values=np.array(vals),
                              base_values=np.array([base_val, base_val]),
                              data=X.iloc[[idx_neg, idx_pos]].values,
                              feature_names=list(features_list))

    # NEGATIVO (classe 0)
    try:
        one = sv[0]
        # se multiclasse binaria: seleziona canale 1 (classe positiva) per coerenza
        try:
            one = sv[0, :, 1]
        except Exception:
            pass
        shap.plots.waterfall(one, show=False)
        plt.title(f"{sheet.capitalize()} – {model_name}\nWaterfall SHAP (individuo NEGATIVO corretto)")
        out1 = os.path.join(subdir, f"{sheet}_{model_tag}_waterfall_negative.png")
        plt.savefig(out1, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[WATERFALL] Salvato: {out1}")
    except Exception as e:
        print(f"[WATERFALL] Errore negativo: {e}")

    # POSITIVO (classe 1)
    try:
        one = sv[1]
        try:
            one = sv[1, :, 1]
        except Exception:
            pass
        shap.plots.waterfall(one, show=False)
        plt.title(f"{sheet.capitalize()} – {model_name}\nWaterfall SHAP (individuo POSITIVO corretto)")
        out2 = os.path.join(subdir, f"{sheet}_{model_tag}_waterfall_positive.png")
        plt.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[WATERFALL] Salvato: {out2}")
    except Exception as e:
        print(f"[WATERFALL] Errore positivo: {e}")

# Per media finale 
final_importance_summary = {metric: {model: [] for model in models} for metric in metrics}

# Per controllare se le feature sono identiche nei due gruppi (serve per media finale)
used_features_by_group = {}

for sheet in ['maschi', 'femmine']:
    print(f"\n======== ANALISI GRUPPO: {sheet.upper()} ========\n")
    df = pd.read_excel(file_path, sheet_name=sheet, header=2, usecols="C:Q")
    df.dropna(inplace=True)
    df['stress'] = df['stress'].astype(int)

    # Lista feature per questo gruppo
    feat_base = features[:] if USE_ONLY[sheet] is None else USE_ONLY[sheet]
    feat_used = [f for f in feat_base if f not in set(EXCLUDE[sheet])]
    used_features_by_group[sheet] = feat_used[:]

    X, y = df[feat_used], df['stress']

    # === BOXPLOT ===
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feat_used):
        plt.subplot(4, 3, i + 1)
        sns.boxplot(data=df, x='stress', y=col, notch=True)
        plt.title(col)
        plt.xlabel('')
        plt.ylabel('')
    plt.suptitle(f"Boxplot – {sheet.upper()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sheet}_boxplot.png"), dpi=300)
    plt.close()
    print(f"Salvato boxplot: {sheet}_boxplot.png")

    importance_summary = {metric: {} for metric in metrics}

    for model_name, clf in models.items():
        print(f"\n--- MODELLO: {model_name} ---")

        # Riallineo X/y difensivo per ogni modello
        X = df[feat_used].reset_index(drop=True).copy()
        y = df['stress'].reset_index(drop=True).copy()
        if len(X) != len(y):
            raise RuntimeError(f"Mismatch X/y: X={len(X)}, y={len(y)} (sheet={sheet}, model={model_name})")

        performance_metrics = []
        conf_matrices = []
        importances_by_metric = {m: [] for m in metrics}
        shap_means_across = []  # mean(|SHAP|) per fold × iter
        # (opzionale) raccolta SHAP std su fold se volessi usarli separatamente
        # shap_stds_across = []

        for _ in range(no_iters):
            y_pred = np.empty(len(X), dtype=int)
            y_proba = np.empty(len(X))

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                clf.fit(X_train, y_train)
                y_pred[test_idx] = clf.predict(X_test)
                y_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

                # === SHAP (TEST o TRAIN secondo flag) ===
                X_eval_for_shap = X_test if USE_TEST_FOR_SHAP else X_train
                shap_means_across.append(shap_mean_abs_binary(clf, X_train, X_eval_for_shap))

                # === Permutation importance: TEST fold se flag True, altrimenti TRAIN fold
                X_perm, y_perm = (X_test, y_test) if USE_TEST_FOR_PERM else (X_train, y_train)
                for metric in metrics:
                    perm = permutation_importance(
                        clf, X_perm, y_perm,
                        scoring=SCORING_MAP[metric], n_repeats=30, random_state=0
                    )
                    importances_by_metric[metric].append(perm.importances_mean)

            # Performance metrics (OOF)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            recall = recall_score(y, y_pred)
            precision = precision_score(y, y_pred)
            roc_auc = roc_auc_score(y, y_proba)
            auprc = average_precision_score(y, y_proba)
            cm = confusion_matrix(y, y_pred)

            performance_metrics.append({
                'accuracy': acc, 'f1': f1, 'recall': recall,
                'precision': precision, 'roc_auc': roc_auc, 'auprc': auprc
            })
            conf_matrices.append(cm)

        means = {k: np.mean([m[k] for m in performance_metrics]) for k in performance_metrics[0]}
        avg_cm = np.round(np.mean(conf_matrices, axis=0)).astype(int)

        print(f"Accuracy:  {means['accuracy']:.4f}")
        print(f"F1-Score:  {means['f1']:.4f}")
        print(f"Recall:    {means['recall']:.4f}")
        print(f"Precision: {means['precision']:.4f}")
        print(f"ROC AUC:   {means['roc_auc']:.4f}")
        print(f"AUPRC:     {means['auprc']:.4f}")
        print(f"Confusion Matrix media:\n{avg_cm}")

        # === SHAP: riepilogo + CSV (top-5) ===
        if len(shap_means_across) > 0:
            shap_arr = np.array(shap_means_across)  # (n_runs, n_features_correnti)
            shap_mean = shap_arr.mean(axis=0)
            shap_std  = shap_arr.std(axis=0)

            order = np.argsort(shap_mean)[::-1]
            print("\n[SHAP] Top-5 mean(|value|) per feature:")
            for i in order[:5]:
                print(f"  - {feat_used[i]:25s} mean={shap_mean[i]:.4f}")

            shap_df = pd.DataFrame({
                "feature": feat_used,
                "shap_mean_abs": shap_mean,
                "shap_std": shap_std
            }).sort_values("shap_mean_abs", ascending=False)
            shap_csv = os.path.join(output_dir, f"{sheet}_{model_name.replace(' ', '_')}_shap_global.csv")
            shap_df.to_csv(shap_csv, index=False)
            print(f"[SHAP] Salvato CSV: {shap_csv}")
        else:
            shap_mean = np.zeros(len(feat_used))
            shap_std  = np.zeros(len(feat_used))
            print("[SHAP] Nessun valore calcolato.")

        # === FOREST PLOT: SOLO su AUPRC (SHAP vs Permutation) ===
        perm_arr = np.array(importances_by_metric['auprc'])  # (n_runs, n_features_correnti)
        if perm_arr.size:
            perm_mean, perm_std = perm_arr.mean(axis=0), perm_arr.std(axis=0)
        else:
            perm_mean = np.zeros(len(feat_used))
            perm_std  = np.zeros(len(feat_used))

        forest_path = os.path.join(
            output_dir,
            f"{sheet}_{model_name.replace(' ', '_')}_forest_SHAP_vs_PERM_auprc.png"
        )
        forest_plot(
            feat_used, perm_mean, perm_std, shap_mean, shap_std,
            title=f"{sheet.capitalize()} – {model_name}: SHAP vs Permutation (AUPRC)",
            outpath=forest_path
        )
        print(f"[FOREST] Salvato: {forest_path}")

        # === Forest plot singoli + normalizzato ===
        # (A) SOLO Permutation (AUPRC)
        perm_only_path = os.path.join(
            output_dir,
            f"{sheet}_{model_name.replace(' ', '_')}_forest_PERM_only_auprc.png"
        )
        forest_plot_single(
            feat_used, perm_mean, perm_std,
            title=f"{sheet.capitalize()} – {model_name}: Permutation importance (AUPRC)",
            outpath=perm_only_path, label="Permutation (AUPRC)"
        )
        print(f"[FOREST] Salvato: {perm_only_path}")

        # (A2) SOLO Permutation NORMALIZZATO
        perm_only_norm = os.path.join(
            output_dir,
            f"{sheet}_{model_name.replace(' ', '_')}_forest_PERM_only_auprc_NORMALIZED.png"
        )
        forest_plot_single_normalized(
            feat_used, perm_mean, perm_std,
            title=f"{sheet.capitalize()} – {model_name}: Permutation (AUPRC) [normalizzato]",
            outpath=perm_only_norm, label="Permutation"
        )
        print(f"[FOREST] Salvato: {perm_only_norm}")

        # (B) SOLO SHAP (global mean |value|)
        shap_only_path = os.path.join(
            output_dir,
            f"{sheet}_{model_name.replace(' ', '_')}_forest_SHAP_only.png"
        )
        forest_plot_single(
            feat_used, shap_mean, shap_std,
            title=f"{sheet.capitalize()} – {model_name}: SHAP global (|mean|)",
            outpath=shap_only_path, label="SHAP |mean|"
        )
        print(f"[FOREST] Salvato: {shap_only_path}")

        # (B2) SOLO SHAP NORMALIZZATO
        shap_only_norm = os.path.join(
            output_dir,
            f"{sheet}_{model_name.replace(' ', '_')}_forest_SHAP_only_NORMALIZED.png"
        )
        forest_plot_single_normalized(
            feat_used, shap_mean, shap_std,
            title=f"{sheet.capitalize()} – {model_name}: SHAP global (|mean|) [normalizzato]",
            outpath=shap_only_norm, label="SHAP |mean|"
        )
        print(f"[FOREST] Salvato: {shap_only_norm}")

        # (C) OVERLAY NORMALIZZATO 
        norm_path = os.path.join(
            output_dir,
            f"{sheet}_{model_name.replace(' ', '_')}_forest_SHAP_vs_PERM_auprc_NORMALIZED.png"
        )
        forest_plot_normalized(
            feat_used, perm_mean, perm_std, shap_mean, shap_std,
            title=f"{sheet.capitalize()} – {model_name}: SHAP vs Permutation (AUPRC) [normalizzato]",
            outpath=norm_path
        )
        print(f"[FOREST] Salvato: {norm_path}")

        # === Media importances (Permutation) ===
        for metric in metrics:
            mean_imp = np.mean(importances_by_metric[metric], axis=0)
            importance_summary[metric][model_name] = mean_imp
            final_importance_summary[metric][model_name].append(mean_imp)

        # ===  Waterfall su 2 individui corretti (uno 0, uno 1) ===
        save_two_waterfalls(clf, X, y, feat_used, sheet, model_name, output_dir)

    # === Grafici RF vs LR (Permutation)===
    for metric in metrics:
        rf_vals = importance_summary[metric]['Random Forest']
        lr_vals = importance_summary[metric]['Logistic Regression']
        x = np.arange(len(feat_used))
        width = 0.35

        print(f"\n>> {sheet.upper()} – {metric.upper()} <<")
        for feat, rf, lr in zip(feat_used, rf_vals, lr_vals):
            print(f"{feat:25s} RF: {rf:.5f} | LR: {lr:.5f}")

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, rf_vals, width, label='Random Forest')
        plt.bar(x + width/2, lr_vals, width, label='Logistic Regression')
        plt.xticks(x, feat_used, rotation=45, ha='right')
        plt.ylabel("Importanza media")
        plt.title(f"{sheet.capitalize()} – Permutation Importance {metric.upper()} RF vs LR")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{sheet}_importance_{metric}_RF_vs_LR.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Salvato grafico: {fname}")

# === GRAFICI FINALI (media su maschi + femmine) ===
groups = list(used_features_by_group.keys())
if used_features_by_group[groups[0]] == used_features_by_group[groups[1]]:
    feat_common = used_features_by_group[groups[0]]
    for metric in metrics:
        rf_vals = np.mean(final_importance_summary[metric]['Random Forest'], axis=0)
        lr_vals = np.mean(final_importance_summary[metric]['Logistic Regression'], axis=0)
        x = np.arange(len(feat_common))
        width = 0.35

        print(f"\n>> MEDIA FINALE – {metric.upper()} <<")
        for feat, rf, lr in zip(feat_common, rf_vals, lr_vals):
            print(f"{feat:25s} RF: {rf:.5f} | LR: {lr:.5f}")

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, rf_vals, width, label='Random Forest')
        plt.bar(x + width/2, lr_vals, width, label='Logistic Regression')
        plt.xticks(x, feat_common, rotation=45, ha='right')
        plt.ylabel("Importanza media")
        plt.title(f"Media finale – Permutation Importance {metric.upper()} RF vs LR")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(output_dir, f"final_average_importance_{metric}_RF_vs_LR.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Salvato grafico medio: {fname}")
else:
    print("\n[INFO] Le feature usate in maschi e femmine sono diverse: salto i grafici di MEDIA FINALE per evitare mismatch.")
