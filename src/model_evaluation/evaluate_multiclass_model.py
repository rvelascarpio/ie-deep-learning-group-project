"""
Comprehensive Multi-Heartbreaker Evaluation with:
- Per-class optimal thresholds (Youden's J statistic)
- Per-class operating points (Sensitivity, Specificity, PPV, NPV, F1)
- Bootstrap 95% confidence intervals for ROC-AUC
- CNN vs LightGBM comparison table
- Updated ROC/PR curves and confusion matrices
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, 
                             average_precision_score, confusion_matrix, 
                             f1_score, precision_score, recall_score)

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def compute_operating_point(y_true, y_prob, threshold):
    """Compute sensitivity, specificity, PPV, NPV, F1 at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1
    }


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for ROC-AUC."""
    rng = np.random.RandomState(seed)
    aucs = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_prob[indices]))
    
    aucs = np.array(aucs)
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lower, upper


def main():
    metadata_path = 'data/subset_multiclass_metadata.csv'
    cnn_probs_path = 'models/clean_oof_multiclass_probs.npy'
    lgbm_probs_path = 'models/lightgbm_oof_multiclass_probs.npy'
    
    df = pd.read_csv(metadata_path)
    
    # Identify which records actually exist on disk and were loaded during training
    base_dir = 'data/raw'
    valid_indices = []
    for i, row in df.iterrows():
        record_path = os.path.join(base_dir, row['filename_lr'])
        if os.path.exists(record_path + '.hea'):
            valid_indices.append(i)
            
    print(f"Total metadata records: {len(df)}")
    print(f"Successfully loaded records on disk: {len(valid_indices)}")
    
    # Filter ground truth labels to valid indices
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    y_true = np.column_stack([df_valid[f'label_{sc}'].values for sc in SUPERCLASSES])
    
    # Load CNN predictions
    has_cnn = os.path.exists(cnn_probs_path)
    if has_cnn:
        cnn_probs = np.load(cnn_probs_path)
        if cnn_probs.shape[0] != len(valid_indices):
            print(f"⚠️  CNN probs size ({cnn_probs.shape[0]}) != valid dataset size ({len(valid_indices)}). CNN comparison skipped.")
            has_cnn = False
        else:
            print("✅ CNN predictions loaded and aligned.")
    
    # Load LightGBM predictions
    has_lgbm = os.path.exists(lgbm_probs_path)
    if has_lgbm:
        lgbm_probs = np.load(lgbm_probs_path)
        if lgbm_probs.shape[0] == len(df):
            # Slice LightGBM predictions to match successfully loaded signals
            lgbm_probs = lgbm_probs[valid_indices]
            print("✅ LightGBM predictions loaded and aligned.")
        elif lgbm_probs.shape[0] != len(valid_indices):
            print(f"⚠️  LightGBM probs size ({lgbm_probs.shape[0]}) mismatch. LightGBM comparison skipped.")
            has_lgbm = False
    
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # =========================================
    # 1. Per-class optimal thresholds & operating points
    # =========================================
    print("\n" + "=" * 60)
    print("PER-CLASS OPTIMAL THRESHOLDS (Youden's J)")
    print("=" * 60)
    
    model_thresholds = {}
    model_operating_points = {}
    
    for model_label, model_probs, available in [("CNN", cnn_probs if has_cnn else None, has_cnn), 
                                                  ("LightGBM", lgbm_probs if has_lgbm else None, has_lgbm)]:
        if not available:
            continue
            
        print("\n" + "-" * 40)
        print(f"Optimal Thresholds for {model_label}")
        print("-" * 40)
        
        thresholds = {}
        operating_points = {}
        for c_idx, sc in enumerate(SUPERCLASSES):
            opt_thresh = find_optimal_threshold(y_true[:, c_idx], model_probs[:, c_idx])
            op = compute_operating_point(y_true[:, c_idx], model_probs[:, c_idx], opt_thresh)
            thresholds[sc] = float(opt_thresh)
            operating_points[sc] = op
            
            print(f"\n[{sc}] Optimal Threshold: {opt_thresh:.3f}")
            print(f"  Sensitivity: {op['sensitivity']:.3f}")
            print(f"  Specificity: {op['specificity']:.3f}")
            print(f"  PPV:         {op['ppv']:.3f}")
            print(f"  NPV:         {op['npv']:.3f}")
            print(f"  F1:          {op['f1']:.3f}")
            
        model_thresholds[model_label] = thresholds
        model_operating_points[model_label] = operating_points
        
        # Save thresholds
        suffix = model_label.lower()
        thresh_path = f'models/multiclass_thresholds_{suffix}.json'
        with open(thresh_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"\nThresholds saved to {thresh_path}")
        
    # For backward compatibility, save LightGBM thresholds as primary (or CNN if LGBM not available)
    primary_suffix = "lightgbm" if has_lgbm else "cnn"
    import shutil
    if os.path.exists(f'models/multiclass_thresholds_{primary_suffix}.json'):
        shutil.copy(f'models/multiclass_thresholds_{primary_suffix}.json', 'models/multiclass_thresholds.json')
    
    # =========================================
    # 2. Bootstrap CIs
    # =========================================
    print("\n" + "=" * 60)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 60)
    
    ci_results = {}
    
    for model_label, model_probs, available in [("CNN", cnn_probs if has_cnn else None, has_cnn), 
                                                  ("LightGBM", lgbm_probs if has_lgbm else None, has_lgbm)]:
        if not available:
            continue
        
        ci_results[model_label] = {}
        print(f"\n--- {model_label} ---")
        for c_idx, sc in enumerate(SUPERCLASSES):
            auc = roc_auc_score(y_true[:, c_idx], model_probs[:, c_idx])
            pr_auc = average_precision_score(y_true[:, c_idx], model_probs[:, c_idx])
            lower, upper = bootstrap_auc_ci(y_true[:, c_idx], model_probs[:, c_idx])
            
            ci_results[model_label][sc] = {
                'roc_auc': auc, 'pr_auc': pr_auc,
                'ci_lower': lower, 'ci_upper': upper
            }
            print(f"  [{sc}] ROC-AUC: {auc:.4f} ({lower:.4f}–{upper:.4f}) | PR-AUC: {pr_auc:.4f}")
    
    # =========================================
    # 3. CNN vs LightGBM comparison table (if both available)
    # =========================================
    if has_cnn and has_lgbm:
        print("\n" + "=" * 60)
        print("CNN vs LightGBM COMPARISON")
        print("=" * 60)
        print(f"{'Class':<8} | {'CNN ROC-AUC':>12} | {'LightGBM ROC-AUC':>16} | {'Winner':>8}")
        print("-" * 55)
        
        for c_idx, sc in enumerate(SUPERCLASSES):
            cnn_auc = ci_results['CNN'][sc]['roc_auc']
            lgbm_auc = ci_results['LightGBM'][sc]['roc_auc']
            winner = "LightGBM" if lgbm_auc > cnn_auc else "CNN"
            print(f"{sc:<8} | {cnn_auc:>12.4f} | {lgbm_auc:>16.4f} | {winner:>8}")
    
    # =========================================
    # 4. Generate plots
    # =========================================
    
    # --- ROC Curves ---
    for model_label, model_probs, available in [("CNN", cnn_probs if has_cnn else None, has_cnn), 
                                                  ("LightGBM", lgbm_probs if has_lgbm else None, has_lgbm)]:
        if not available:
            continue
        
        plt.figure(figsize=(10, 8))
        for c_idx, (sc, color) in enumerate(zip(SUPERCLASSES, COLORS)):
            fpr, tpr, _ = roc_curve(y_true[:, c_idx], model_probs[:, c_idx])
            auc = roc_auc_score(y_true[:, c_idx], model_probs[:, c_idx])
            lower, upper = bootstrap_auc_ci(y_true[:, c_idx], model_probs[:, c_idx])
            plt.plot(fpr, tpr, color=color, lw=2.5, 
                     label=f'{sc} (AUC = {auc:.3f}, 95% CI: {lower:.3f}–{upper:.3f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title(f'Multi-Heartbreaker {model_label} — ROC Curves with 95% CI', fontsize=15, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        suffix = model_label.lower()
        roc_path = f'reports/figures/multiclass_roc_curve_{suffix}.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {roc_path}")
    
    # --- PR Curves ---
    for model_label, model_probs, available in [("CNN", cnn_probs if has_cnn else None, has_cnn), 
                                                  ("LightGBM", lgbm_probs if has_lgbm else None, has_lgbm)]:
        if not available:
            continue
        
        plt.figure(figsize=(10, 8))
        for c_idx, (sc, color) in enumerate(zip(SUPERCLASSES, COLORS)):
            precision, recall, _ = precision_recall_curve(y_true[:, c_idx], model_probs[:, c_idx])
            pr_auc = average_precision_score(y_true[:, c_idx], model_probs[:, c_idx])
            plt.plot(recall, precision, color=color, lw=2.5, label=f'{sc} (PR-AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=13)
        plt.ylabel('Precision (PPV)', fontsize=13)
        plt.title(f'Multi-Heartbreaker {model_label} — Precision-Recall Curves', fontsize=15, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        
        suffix = model_label.lower()
        pr_path = f'reports/figures/multiclass_pr_curve_{suffix}.png'
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {pr_path}")
    
    # --- Confusion Matrices (using per-class thresholds) ---
    for model_label, model_probs, available in [("CNN", cnn_probs if has_cnn else None, has_cnn), 
                                                  ("LightGBM", lgbm_probs if has_lgbm else None, has_lgbm)]:
        if not available:
            continue
            
        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        axes = axes.flatten()
        
        thresholds = model_thresholds[model_label]
        operating_points = model_operating_points[model_label]
        
        for c_idx, sc in enumerate(SUPERCLASSES):
            thresh = thresholds.get(sc, 0.5)
            y_pred = (model_probs[:, c_idx] >= thresh).astype(int)
            cm = confusion_matrix(y_true[:, c_idx], y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[c_idx],
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'],
                        annot_kws={"size": 14})
            
            op = operating_points.get(sc, {})
            title = f'{sc} (t={thresh:.2f})\nSens={op.get("sensitivity", 0):.2f} Spec={op.get("specificity", 0):.2f} F1={op.get("f1", 0):.2f}'
            axes[c_idx].set_title(title, fontsize=12, fontweight='bold')
            axes[c_idx].set_ylabel('True Label', fontsize=11)
            axes[c_idx].set_xlabel('Predicted Label', fontsize=11)
        
        fig.delaxes(axes[-1])
        plt.suptitle(f'Multi-Heartbreaker {model_label} — Per-Class Confusion Matrices\n(Optimal Thresholds via Youden\'s J)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        suffix = model_label.lower()
        cm_path = f'reports/figures/multiclass_confusion_matrix_{suffix}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {cm_path}")
    
    # --- Operating Points Summary Table (as image) ---
    for model_label, model_probs, available in [("CNN", cnn_probs if has_cnn else None, has_cnn), 
                                                  ("LightGBM", lgbm_probs if has_lgbm else None, has_lgbm)]:
        if not available:
            continue
            
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        
        thresholds = model_thresholds[model_label]
        operating_points = model_operating_points[model_label]
        
        table_data = []
        for sc in SUPERCLASSES:
            op = operating_points[sc]
            ci = ci_results.get(model_label, {}).get(sc, {})
            table_data.append([
                sc,
                f"{op['threshold']:.3f}",
                f"{ci.get('roc_auc', 0):.4f} ({ci.get('ci_lower', 0):.4f}–{ci.get('ci_upper', 0):.4f})",
                f"{op['sensitivity']:.3f}",
                f"{op['specificity']:.3f}",
                f"{op['ppv']:.3f}",
                f"{op['npv']:.3f}",
                f"{op['f1']:.3f}"
            ])
        
        col_labels = ['Class', 'Threshold', 'ROC-AUC (95% CI)', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
        table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)
        
        # Style header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#2196F3')
            table[0, j].set_text_props(color='white', fontweight='bold')
        
        plt.title(f'Multi-Heartbreaker {model_label} — Per-Class Operating Points', fontsize=14, fontweight='bold', pad=20)
        suffix = model_label.lower()
        op_path = f'reports/figures/multiclass_operating_points_{suffix}.png'
        plt.savefig(op_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {op_path}")
    
    # Copy primary outputs to default filenames for backward compatibility
    primary_suffix = "lightgbm" if has_lgbm else "cnn"
    shutil.copy(f'reports/figures/multiclass_confusion_matrix_{primary_suffix}.png', 'reports/figures/multiclass_confusion_matrix.png')
    shutil.copy(f'reports/figures/multiclass_operating_points_{primary_suffix}.png', 'reports/figures/multiclass_operating_points.png')
    
    # Also save the old-style combined ROC/PR as multiclass_roc_curve.png for backward compat
    shutil.copy(f'reports/figures/multiclass_roc_curve_{primary_suffix}.png', 'reports/figures/multiclass_roc_curve.png')
    shutil.copy(f'reports/figures/multiclass_pr_curve_{primary_suffix}.png', 'reports/figures/multiclass_pr_curve.png')
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
