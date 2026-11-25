"""
train_models.py
----------------
Trains and compares a broad suite of classification algorithms for
attrition prediction, handles class imbalance, tunes the strongest
candidates, evaluates with a full metric suite + diagnostic plots,
generates SHAP explainability artifacts, and versions the winning
model with full metadata for reproducible deployment.

Run: python train_models.py
"""

import json
import os
import time
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import EVAL_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, get_logger
from preprocessing import get_processed_splits

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ------------------------------------------------------------------
# Model zoo
# ------------------------------------------------------------------
def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": __import__("sklearn.tree", fromlist=["DecisionTreeClassifier"]).DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           eval_metric="logloss", random_state=RANDOM_STATE,
                                           scale_pos_weight=5, n_jobs=-1)
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                             class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                                  auto_class_weights="Balanced", random_state=RANDOM_STATE,
                                                  verbose=False)
    return models


def add_ensembles(models: dict) -> dict:
    """Add Voting + Stacking ensembles built from the strongest base learners."""
    estimators = [
        ("rf", models["Random Forest"]),
        ("gb", models["Gradient Boosting"]),
        ("lr", models["Logistic Regression"]),
    ]
    if HAS_XGB:
        estimators.append(("xgb", models["XGBoost"]))

    models["Voting Ensemble"] = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    models["Stacking Ensemble"] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    return models


# ------------------------------------------------------------------
# Class imbalance handling
# ------------------------------------------------------------------
def resample_training_data(X_train, y_train, method: str = "smote"):
    if not HAS_IMBLEARN or method == "none":
        return X_train, y_train
    samplers = {
        "smote": SMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
        "borderline_smote": BorderlineSMOTE(random_state=RANDOM_STATE),
    }
    sampler = samplers.get(method)
    if sampler is None:
        return X_train, y_train
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.info(f"Resampled with {method}: {X_train.shape[0]} -> {X_res.shape[0]} rows "
                f"(attrition rate {y_res.mean():.2%})")
    return X_res, y_res


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    pr_auc = trapz_fn(recalls[::-1], precisions[::-1])

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": float(pr_auc),
        "log_loss": log_loss(y_test, y_proba),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "specificity": specificity,
        "sensitivity": sensitivity,
    }


def plot_diagnostics(model, model_name: str, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Stayed", "Left"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"cm_{safe_name}.png"), dpi=110)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}", color="#0277BD")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"roc_{safe_name}.png"), dpi=110)
    plt.close()

    # Precision-Recall curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4.5))
    plt.plot(recalls, precisions, color="#6A1B9A")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"pr_{safe_name}.png"), dpi=110)
    plt.close()

    # Calibration curve
    try:
        frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
        plt.figure(figsize=(5, 4.5))
        plt.plot(mean_pred, frac_pos, marker="o", color="#EF6C00", label=model_name)
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability"); plt.ylabel("Observed Frequency")
        plt.title(f"Calibration Curve — {model_name}"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f"calibration_{safe_name}.png"), dpi=110)
        plt.close()
    except Exception as e:
        logger.warning(f"Calibration curve failed for {model_name}: {e}")

    # Learning curve
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=3, scoring="roc_auc",
            train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
        )
        plt.figure(figsize=(5.5, 4.5))
        plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train", color="#2E7D32")
        plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation", color="#C62828")
        plt.xlabel("Training examples"); plt.ylabel("ROC AUC")
        plt.title(f"Learning Curve — {model_name}"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f"learning_curve_{safe_name}.png"), dpi=110)
        plt.close()
    except Exception as e:
        logger.warning(f"Learning curve failed for {model_name}: {e}")


# ------------------------------------------------------------------
# Hyperparameter tuning (RandomizedSearchCV) for top candidates
# ------------------------------------------------------------------
TUNING_GRIDS = {
    "Random Forest": {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [150, 250, 350, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    },
    "LightGBM": {
        "n_estimators": [150, 250, 350, 500],
        "max_depth": [4, 6, 8, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 0.5, 1, 5, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
}


def tune_model(name, model, X_train, y_train, n_iter=15):
    grid = TUNING_GRIDS.get(name)
    if grid is None:
        return model, {}
    logger.info(f"Tuning {name} with RandomizedSearchCV ({n_iter} iterations)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(model, grid, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    logger.info(f"  Best params for {name}: {search.best_params_} (CV ROC AUC={search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_


# ------------------------------------------------------------------
# Explainability (SHAP)
# ------------------------------------------------------------------
def generate_shap_artifacts(model, model_name, X_train, X_test, feature_names):
    if not HAS_SHAP:
        logger.warning("SHAP not installed — skipping explainability artifacts.")
        return
    try:
        if hasattr(model, "predict_proba") and model_name in (
            "Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Decision Tree", "AdaBoost"
        ):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            background = shap.sample(X_train, min(100, len(X_train)), random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test.iloc[:100], nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            X_test = X_test.iloc[:100]

        # Global summary (bar)
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Global Feature Importance — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, "shap_global_importance.png"), dpi=120)
        plt.close()

        # Beeswarm summary
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, "shap_beeswarm.png"), dpi=120)
        plt.close()

        # Save raw SHAP values + a few single-employee explanations
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}).sort_values(
            "mean_abs_shap", ascending=False)
        shap_df.to_csv(os.path.join(OUTPUTS_DIR, "shap_feature_importance.csv"), index=False)

        joblib.dump({"shap_values": shap_values[:200], "X_test_sample": X_test.iloc[:200]},
                    os.path.join(MODELS_DIR, "shap_explainer_cache.pkl"))

        logger.info("Saved SHAP global importance, beeswarm plot, and feature importance CSV.")
    except Exception as e:
        logger.warning(f"SHAP explainability generation failed: {e}")


# ------------------------------------------------------------------
# Main training orchestration
# ------------------------------------------------------------------
def train_and_evaluate(resample_method: str = "smote", tune_top_n: int = 3):
    X_train, X_test, y_train, y_test, feature_names, encoders, scaler = get_processed_splits()

    models = get_models()
    models = add_ensembles(models)

    X_train_res, y_train_res = resample_training_data(X_train, y_train, method=resample_method)

    results, trained_models, timings = {}, {}, {}

    logger.info("=" * 70)
    logger.info(f"TRAINING {len(models)} CLASSIFICATION MODELS")
    logger.info("=" * 70)

    for name, model in models.items():
        t0 = time.time()
        try:
            model.fit(X_train_res, y_train_res)
        except Exception as e:
            logger.warning(f"Skipping {name}: {e}")
            continue
        elapsed = time.time() - t0
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model
        timings[name] = elapsed
        logger.info(f"{name:22s} | ROC AUC={metrics['roc_auc']:.4f} | Recall={metrics['recall']:.4f} "
                    f"| F1={metrics['f1_score']:.4f} | ({elapsed:.1f}s)")

    # --- Hyperparameter tuning for the top-N models by ROC AUC ---
    leaderboard = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    top_candidates = [m for m in leaderboard.index if m in TUNING_GRIDS][:tune_top_n]

    for name in top_candidates:
        fresh_models = get_models()
        base_model = fresh_models[name] if name in fresh_models else add_ensembles(fresh_models)[name]
        tuned_model, best_params = tune_model(name, base_model, X_train_res, y_train_res)
        metrics = evaluate_model(tuned_model, X_test, y_test)
        tuned_name = f"{name} (Tuned)"
        results[tuned_name] = metrics
        trained_models[tuned_name] = tuned_model
        logger.info(f"{tuned_name:22s} | ROC AUC={metrics['roc_auc']:.4f} | Recall={metrics['recall']:.4f}")

    # --- Final leaderboard ---
    leaderboard = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    leaderboard.index.name = "model"
    leaderboard.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))
    leaderboard.round(4).to_json(os.path.join(OUTPUTS_DIR, "model_leaderboard.json"), orient="index", indent=2)

    best_model_name = leaderboard.index[0]
    best_model = trained_models[best_model_name]

    logger.info("=" * 70)
    logger.info(f"BEST MODEL: {best_model_name} (ROC AUC = {leaderboard.loc[best_model_name, 'roc_auc']:.4f})")
    logger.info("=" * 70)

    # --- Diagnostics for best model + top 3 ---
    for name in leaderboard.index[:3]:
        plot_diagnostics(trained_models[name], name, X_test, y_test, X_train_res, y_train_res)

    # --- Feature importance ---
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": feature_names, "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
    elif hasattr(best_model, "coef_"):
        feature_importance = pd.DataFrame({
            "feature": feature_names, "importance": np.abs(best_model.coef_[0])
        }).sort_values("importance", ascending=False)

    if feature_importance is not None:
        feature_importance.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance.csv"), index=False)
        plt.figure(figsize=(8, 7))
        feature_importance.head(20).sort_values("importance").plot(
            x="feature", y="importance", kind="barh", legend=False, color="#0277BD", ax=plt.gca())
        plt.title(f"Top 20 Feature Importances — {best_model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, "feature_importance_top20.png"), dpi=120)
        plt.close()

    # --- Permutation importance (model-agnostic sanity check) ---
    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                                       random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
        perm_df = pd.DataFrame({"feature": feature_names, "perm_importance_mean": perm.importances_mean,
                                 "perm_importance_std": perm.importances_std}).sort_values(
            "perm_importance_mean", ascending=False)
        perm_df.to_csv(os.path.join(OUTPUTS_DIR, "permutation_importance.csv"), index=False)
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")

    # --- SHAP explainability ---
    generate_shap_artifacts(best_model, best_model_name.replace(" (Tuned)", ""), X_train_res, X_test, feature_names)

    # --- Save best model + versioned metadata ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"best_model_{timestamp}.pkl"))

    metadata = {
        "model_name": best_model_name,
        "metrics": results[best_model_name],
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "trained_at": timestamp,
        "resample_method": resample_method,
        "version": "2.0.0",
        "dataset_rows": int(X_train.shape[0] + X_test.shape[0]),
    }
    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved best model -> {MODELS_DIR}/best_model.pkl (versioned copy: best_model_{timestamp}.pkl)")
    logger.info(f"Saved leaderboard -> {OUTPUTS_DIR}/model_comparison.csv")

    print("\n" + "=" * 70)
    print("MODEL LEADERBOARD (sorted by ROC AUC)")
    print("=" * 70)
    print(leaderboard[["roc_auc", "recall", "precision", "f1_score", "balanced_accuracy"]].round(4).to_string())

    return best_model, best_model_name, results, feature_names, encoders, scaler


if __name__ == "__main__":
    train_and_evaluate()
