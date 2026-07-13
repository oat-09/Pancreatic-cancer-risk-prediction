"""
=============================================================================
  Pancreatic Cancer Risk Prediction — Wearable-Compatible ML Pipeline (v2)
=============================================================================
  Project : Multi-Modal ML Framework for Early Pancreatic Cancer Risk
            Stratification Using Wearable Physiological Monitoring
  Group   : G-39
  Guide   : Dr. V. T. Lokare
  Dept.   : Computer Science & Engineering
  College : Rajarambapu Institute of Technology, Rajaramnagar

  Description
  -----------
  End-to-end ML pipeline that trains and compares seven classifiers on
  clinical/metabolic features compatible with a future wearable (garment
  based) sensor platform, tunes the best-performing model, evaluates it
  with the metric suite expected by reviewers (ROC, PR, calibration,
  ablation, SHAP), and packages it for deployment.

  Predicts pancreatic cancer risk across three classes:
      0 — Healthy
      1 — Benign (non-malignant pancreatic condition)
      2 — PDAC   (Pancreatic Ductal Adenocarcinoma)

  Wearable Note
  -------------
  Features such as real-time glucose, sweat pH, and sweat conductivity will
  be streamed from the ESP32-based garment sensor during deployment. They
  are NOT included in training due to dataset limitations, but the
  pipeline is designed to accept them seamlessly at inference time.

  What's new vs. v1
  ------------------
    - Multi-model comparison   : RF, Extra Trees, XGBoost/GB, LightGBM/HGB,
                                  SVM, Decision Tree, Logistic Regression
    - Hyperparameter tuning    : RandomizedSearchCV on the CV-selected winner
    - Feature engineering      : symptom_score, lifestyle_risk_score
    - Ablation study           : incremental feature-group contribution
    - Explainability           : SHAP summary + bar plot (if shap installed)
    - Extra diagnostics        : PR curve, calibration curve, correlation
                                  heatmap, class-distribution plot
    - Repeated Stratified K-Fold CV instead of a single 5-fold pass

  Requirements
  ------------
      pip install scikit-learn pandas numpy matplotlib seaborn joblib
      # Optional, auto-detected — pipeline degrades gracefully without them:
      pip install xgboost lightgbm shap
=============================================================================
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Third-party (core) ──────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble          import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.svm               import SVC
from sklearn.tree              import DecisionTreeClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute            import SimpleImputer
from sklearn.model_selection   import (
    train_test_split, cross_validate, RandomizedSearchCV,
    StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.calibration       import calibration_curve
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import joblib

# ─── Optional third-party (auto-detected) ────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "pancreatic_final_dataset.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "model.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────
CLASS_NAMES   = ["Healthy", "Benign", "PDAC"]
CLASS_COLORS  = ["#378ADD", "#EF9F27", "#1D9E75"]
FIG_DPI       = 150
sns.set_theme(style="whitegrid", font_scale=1.0)

print("\n" + "=" * 70)
print("  PANCREATIC CANCER RISK PREDICTION — ML PIPELINE (v2)")
print("=" * 70)
print(f"  XGBoost  available : {XGBOOST_AVAILABLE}")
print(f"  LightGBM available : {LIGHTGBM_AVAILABLE}")
print(f"  SHAP     available : {SHAP_AVAILABLE}")
print("=" * 70)

# =============================================================================
# SECTION 1 — LOAD DATASET
# =============================================================================
print("\n[STEP 1]  Loading dataset …")
df_raw = pd.read_csv(DATA_PATH)
print(f"          Rows    : {df_raw.shape[0]:,}")
print(f"          Columns : {df_raw.shape[1]}")
print(f"          Classes : {df_raw['diagnosis'].value_counts().sort_index().to_dict()}")

# =============================================================================
# SECTION 2 — FEATURE SELECTION (wearable-compatible clinical features)
# =============================================================================
print("\n[STEP 2]  Selecting wearable-compatible clinical features …")

# Features selected because they are:
#   (a) Directly obtainable from wearable sensors or paired app questionnaire
#   (b) Clinically validated risk indicators for pancreatic cancer
#   (c) Free of PII (no ID, Name, or timestamps)
#
# NOT included (wearable integration planned for deployment phase):
#   - sweat_pH, sweat_conductivity  → ESP32 + pH sensor
#   - real-time ISF glucose         → continuous glucose sensor patch
#   - heart_rate, skin_temperature  → garment-embedded sensors

BASE_FEATURES = [
    "age",                    # entered via companion app
    "sex",                    # entered via companion app (0=Female, 1=Male)
    "glucose",                # blood/ISF glucose — future: wearable real-time
    "smoking_history",        # app questionnaire  (0=No, 1=Yes)
    "diabetes_history",       # app questionnaire  (0=No, 1=Yes)
    "family_history",         # app questionnaire  (0=No, 1=Yes)
    "chronic_pancreatitis",   # clinical intake    (0=No, 1=Yes)
    "jaundice",               # symptom flag       (0=No, 1=Yes)
    "weight_loss",            # symptom flag       (0=No, 1=Yes)
    "abdominal_pain",         # symptom flag       (0=No, 1=Yes)
    "back_pain",              # symptom flag       (0=No, 1=Yes)
]
TARGET = "diagnosis"

# Feature groups — reused later for the ablation study
DEMOGRAPHIC_FEATURES = ["age", "sex"]
LIFESTYLE_FEATURES   = ["smoking_history", "diabetes_history", "family_history"]
SYMPTOM_FEATURES     = ["jaundice", "weight_loss", "abdominal_pain", "back_pain", "chronic_pancreatitis"]
BIOMARKER_FEATURES   = ["glucose"]
ENGINEERED_FEATURES  = ["symptom_score", "lifestyle_risk_score"]

print(f"          Base features selected : {len(BASE_FEATURES)}")
for f in BASE_FEATURES:
    print(f"            - {f}")

# =============================================================================
# SECTION 3 — PREPROCESSING (impute + encode on the base features)
# =============================================================================
print("\n[STEP 3]  Preprocessing …")

df = df_raw[BASE_FEATURES + [TARGET]].copy()

print("          Missing values before imputation :")
missing = df[BASE_FEATURES].isnull().sum()
for col, cnt in missing[missing > 0].items():
    pct = cnt / len(df) * 100
    print(f"            {col:<25} {cnt:>5} ({pct:.1f}%)")

# Continuous features  → median  (robust to outliers)
# Binary/flag features → mode    (most common value)
continuous_cols = ["age", "glucose"]
binary_cols     = [c for c in BASE_FEATURES if c not in continuous_cols]

cont_imputer   = SimpleImputer(strategy="median")
binary_imputer = SimpleImputer(strategy="most_frequent")

df[continuous_cols] = cont_imputer.fit_transform(df[continuous_cols])
df[binary_cols]     = binary_imputer.fit_transform(df[binary_cols])

print(f"\n          Missing values after imputation : {df.isnull().sum().sum()}")

le_dict = {}
for col in BASE_FEATURES:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        print(f"          Label-encoded: {col}")

assert df[BASE_FEATURES].select_dtypes(exclude=[np.number]).empty, \
    "Non-numeric columns remain — check encoding step."
print("          All base features confirmed numeric.")

print("\n          Class distribution :")
for cls, cnt in df[TARGET].value_counts().sort_index().items():
    print(f"            {CLASS_NAMES[cls]:<10} {cnt:>5} samples")

# =============================================================================
# SECTION 4 — FEATURE ENGINEERING
# =============================================================================
print("\n[STEP 4]  Engineering composite risk-score features …")

# Symptom score: count of active PDAC-associated symptoms (0-5)
df["symptom_score"] = (
    df["jaundice"] + df["weight_loss"] +
    df["abdominal_pain"] + df["back_pain"] + df["chronic_pancreatitis"]
)

# Lifestyle risk score: count of active modifiable/hereditary risk factors (0-3)
df["lifestyle_risk_score"] = (
    df["smoking_history"] + df["diabetes_history"] + df["family_history"]
)

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
print(f"          Added : symptom_score (0-5), lifestyle_risk_score (0-3)")
print(f"          Total feature count now : {len(ALL_FEATURES)}")

# =============================================================================
# SECTION 5 — EXPLORATORY PLOTS (class distribution + correlation heatmap)
# =============================================================================
print("\n[STEP 5]  Generating exploratory plots …")

# ── Plot: class distribution ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.5))
counts = df[TARGET].value_counts().sort_index()
ax.bar(CLASS_NAMES, counts.values, color=CLASS_COLORS, edgecolor="white")
for i, v in enumerate(counts.values):
    ax.text(i, v + max(counts.values) * 0.01, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Class Distribution — Diagnosis Labels", fontsize=12)
ax.set_ylabel("Number of samples", fontsize=10)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(dist_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"          Saved → {dist_path}")

# ── Plot: correlation heatmap ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 7.0))
corr = df[ALL_FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.4, annot_kws={"size": 7}, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=12, pad=10)
plt.tight_layout()
corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(corr_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"          Saved → {corr_path}")

# =============================================================================
# SECTION 6 — TRAIN / TEST SPLIT  (80 / 20, stratified)
# =============================================================================
print("\n[STEP 6]  Splitting dataset (80% train / 20% test, stratified) …")

X = df[ALL_FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    stratify     = y,
    random_state = RANDOM_STATE
)

print(f"          Training samples : {len(y_train):,}")
print(f"          Testing  samples : {len(y_test):,}")

# =============================================================================
# SECTION 7 — MODEL COMPARISON
# =============================================================================
print("\n[STEP 7]  Training and comparing candidate models …")

def build_models():
    """Return an ordered dict of name -> sklearn Pipeline (scaler + estimator).
    Scaling is harmless for tree ensembles and required for SVM/Logistic
    Regression, so a single Pipeline definition keeps the comparison fair
    and simple to extend."""
    models = {}

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=150, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models["Extra Trees"] = ExtraTreesClassifier(
        n_estimators=150, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        # Graceful substitute if xgboost isn't installed in this environment
        models["Gradient Boosting (XGBoost fallback)"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08,
            random_state=RANDOM_STATE
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.08,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        )
    else:
        # Graceful substitute if lightgbm isn't installed in this environment
        models["Hist Gradient Boosting (LightGBM fallback)"] = HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.08, random_state=RANDOM_STATE
        )

    models["SVM (RBF)"] = SVC(
        kernel="rbf", probability=True, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=8, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    models["Logistic Regression"] = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
    )
    return models

def make_pipeline(estimator):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator)
    ])

raw_models   = build_models()
rskf         = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
comparison   = []
fitted_pipes = {}

for name, estimator in raw_models.items():
    pipe = make_pipeline(estimator)
    cv_res = cross_validate(
        pipe, X_train, y_train, cv=rskf,
        scoring=["accuracy", "f1_macro"], n_jobs=-1
    )
    pipe.fit(X_train, y_train)
    y_pred_i  = pipe.predict(X_test)
    y_proba_i = pipe.predict_proba(X_test)
    y_bin_i   = label_binarize(y_test, classes=[0, 1, 2])

    row = {
        "Model":           name,
        "CV Accuracy":     cv_res["test_accuracy"].mean(),
        "CV Accuracy Std": cv_res["test_accuracy"].std(),
        "CV F1 (macro)":   cv_res["test_f1_macro"].mean(),
        "Test Accuracy":   accuracy_score(y_test, y_pred_i),
        "Test Precision":  precision_score(y_test, y_pred_i, average="macro", zero_division=0),
        "Test Recall":     recall_score(y_test, y_pred_i, average="macro"),
        "Test F1 (macro)": f1_score(y_test, y_pred_i, average="macro"),
        "Test ROC-AUC":    roc_auc_score(y_bin_i, y_proba_i, multi_class="ovr", average="macro"),
    }
    comparison.append(row)
    fitted_pipes[name] = pipe
    print(f"          {name:<38} CV F1={row['CV F1 (macro)']:.4f}  "
          f"Test Acc={row['Test Accuracy']:.4f}  Test AUC={row['Test ROC-AUC']:.4f}")

comparison_df = pd.DataFrame(comparison).sort_values("CV F1 (macro)", ascending=False).reset_index(drop=True)
comparison_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
comparison_df.to_csv(comparison_csv, index=False)
print(f"\n          Comparison table saved → {comparison_csv}")
print(comparison_df.round(4).to_string(index=False))

# ── Plot: model comparison bar chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5.5))
plot_df = comparison_df.set_index("Model")[["Test Accuracy", "Test F1 (macro)", "Test ROC-AUC"]]
plot_df.plot.bar(ax=ax, color=["#378ADD", "#EF9F27", "#1D9E75"], edgecolor="white", width=0.75)
ax.set_title("Model Comparison — Hold-out Test Set", fontsize=12)
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc="lower right")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
model_cmp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(model_cmp_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"          Saved → {model_cmp_path}")

# =============================================================================
# SECTION 8 — HYPERPARAMETER TUNING (winner from CV F1 ranking)
# =============================================================================
BEST_MODEL_NAME = comparison_df.iloc[0]["Model"]
print(f"\n[STEP 8]  Tuning hyperparameters for the CV winner: {BEST_MODEL_NAME} …")

PARAM_DISTRIBUTIONS = {
    "Random Forest": {
        "clf__n_estimators":      [100, 150, 200, 300, 400],
        "clf__max_depth":         [None, 6, 10, 14, 20],
        "clf__min_samples_split": [2, 5, 8, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2", None],
    },
    "Extra Trees": {
        "clf__n_estimators":      [100, 150, 200, 300, 400],
        "clf__max_depth":         [None, 6, 10, 14, 20],
        "clf__min_samples_split": [2, 5, 8, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2", None],
    },
    "XGBoost": {
        "clf__n_estimators":  [100, 200, 300, 400],
        "clf__max_depth":     [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.2],
        "clf__subsample":     [0.7, 0.8, 0.9, 1.0],
    },
    "Gradient Boosting (XGBoost fallback)": {
        "clf__n_estimators":  [100, 200, 300],
        "clf__max_depth":     [2, 3, 4, 5],
        "clf__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.2],
    },
    "LightGBM": {
        "clf__n_estimators":  [100, 200, 300, 400],
        "clf__max_depth":     [-1, 4, 6, 8, 10],
        "clf__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.2],
        "clf__num_leaves":    [15, 31, 63, 127],
    },
    "Hist Gradient Boosting (LightGBM fallback)": {
        "clf__max_depth":     [None, 4, 6, 8, 10],
        "clf__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.2],
        "clf__max_iter":      [100, 150, 200, 300],
    },
    "SVM (RBF)": {
        "clf__C":     [0.1, 1, 3, 10, 30],
        "clf__gamma": ["scale", "auto", 0.01, 0.1],
    },
    "Decision Tree": {
        "clf__max_depth":         [3, 5, 8, 12, None],
        "clf__min_samples_split": [2, 5, 8, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
    },
    "Logistic Regression": {
        "clf__C":       [0.01, 0.1, 1, 3, 10],
        "clf__penalty": ["l2"],
        "clf__solver":  ["lbfgs"],
    },
}

base_pipe   = make_pipeline(build_models()[BEST_MODEL_NAME])
param_dist  = PARAM_DISTRIBUTIONS.get(BEST_MODEL_NAME, {})
tuning_cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

if param_dist:
    search = RandomizedSearchCV(
        base_pipe, param_distributions=param_dist,
        n_iter=25, scoring="f1_macro", cv=tuning_cv,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"          Best CV F1 (macro) after tuning : {search.best_score_:.4f}")
    print(f"          Best params :")
    for k, v in search.best_params_.items():
        print(f"            {k} = {v}")
else:
    print("          No tuning grid defined for this model — using default hyperparameters.")
    best_model = fitted_pipes[BEST_MODEL_NAME]
    best_model.fit(X_train, y_train)

FINAL_MODEL_NAME = BEST_MODEL_NAME

# =============================================================================
# SECTION 9 — FINAL MODEL EVALUATION
# =============================================================================
print(f"\n[STEP 9]  Evaluating tuned final model ({FINAL_MODEL_NAME}) on hold-out test set …")

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)
y_bin   = label_binarize(y_test, classes=[0, 1, 2])

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro")
f1   = f1_score(y_test, y_pred, average="macro")
roc_auc_macro = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

print("\n" + "-" * 45)
print(f"  Accuracy          : {acc   * 100:.2f}%")
print(f"  Macro Precision   : {prec  * 100:.2f}%")
print(f"  Macro Recall      : {rec   * 100:.2f}%")
print(f"  Macro F1-score    : {f1    * 100:.2f}%")
print(f"  ROC-AUC (macro)   : {roc_auc_macro:.4f}")
print("-" * 45)

print("\n  Classification Report :")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print("  Per-class ROC-AUC :")
for i, cls in enumerate(CLASS_NAMES):
    auc_val = roc_auc_score(y_bin[:, i], y_proba[:, i])
    print(f"    {cls:<10} : {auc_val:.4f}")

# ── Plot 1 : Confusion Matrix ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.0))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, annot_kws={"size": 14, "weight": "bold"}, ax=ax)
ax.set_title(f"Confusion Matrix — {FINAL_MODEL_NAME} (tuned)", fontsize=12, pad=12)
ax.set_ylabel("True Label", fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"\n          Saved → {cm_path}")

# ── Plot 2 : ROC Curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 5.5))
line_styles = ["-", "--", "-."]
for i, (cls, col, ls) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, line_styles)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    roc_val     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2.2, ls=ls, label=f"{cls}  (AUC = {roc_val:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.45, label="Random classifier (AUC = 0.5)")
ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.set_title(f"ROC Curves — One-vs-Rest per Class (Macro AUC = {roc_auc_macro:.4f})", fontsize=12)
ax.legend(fontsize=10, loc="lower right")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"          Saved → {roc_path}")

# ── Plot 3 : Precision-Recall Curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 5.5))
for i, (cls, col, ls) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, line_styles)):
    p, r, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
    ap = average_precision_score(y_bin[:, i], y_proba[:, i])
    ax.plot(r, p, color=col, lw=2.2, ls=ls, label=f"{cls}  (AP = {ap:.4f})")
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.02])
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves — One-vs-Rest per Class", fontsize=12)
ax.legend(fontsize=10, loc="lower left")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
pr_path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
plt.savefig(pr_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"          Saved → {pr_path}")

# ── Plot 4 : Calibration Curves ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for i, (cls, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    frac_pos, mean_pred = calibration_curve(y_bin[:, i], y_proba[:, i], n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", color=col, lw=1.8, label=cls)
ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Observed frequency", fontsize=11)
ax.set_title("Calibration Curves — One-vs-Rest per Class", fontsize=12)
ax.legend(fontsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
calib_path = os.path.join(OUTPUT_DIR, "calibration_curve.png")
plt.savefig(calib_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"          Saved → {calib_path}")

# ── Plot 5 : Feature Importance (tree-based models only) ─────────────────────
fi_path = None
final_estimator = best_model.named_steps["clf"]
if hasattr(final_estimator, "feature_importances_"):
    fi = pd.Series(final_estimator.feature_importances_, index=ALL_FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = ["#378ADD" if v >= fi.median() else "#B4B2A9" for v in fi]
    fi.plot.barh(ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Feature Importance — {FINAL_MODEL_NAME} (tuned)", fontsize=12)
    ax.set_xlabel("Importance score", fontsize=10)
    ax.axvline(fi.median(), color="crimson", ls="--", lw=1.2, label=f"Median ({fi.median():.4f})")
    ax.legend(fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"          Saved → {fi_path}")
else:
    print("          Skipped feature-importance plot (model has no native importances).")

# =============================================================================
# SECTION 10 — SHAP EXPLAINABILITY (optional)
# =============================================================================
shap_summary_path = None
print("\n[STEP 10]  SHAP explainability …")
if SHAP_AVAILABLE:
    try:
        X_test_df = pd.DataFrame(X_test, columns=ALL_FEATURES)
        if hasattr(final_estimator, "feature_importances_"):
            explainer   = shap.TreeExplainer(final_estimator)
            X_shap_in   = best_model.named_steps["scaler"].transform(X_test)
            shap_values = explainer.shap_values(X_shap_in)
        else:
            # Kernel/linear fallback for non-tree models — small background sample for speed
            background  = shap.sample(best_model.named_steps["scaler"].transform(X_train), 100, random_state=RANDOM_STATE)
            explainer   = shap.KernelExplainer(final_estimator.predict_proba, background)
            X_shap_in   = best_model.named_steps["scaler"].transform(X_test[:150])
            shap_values = explainer.shap_values(X_shap_in)
            X_test_df   = X_test_df.iloc[:150]

        # shap_values is a list per class for multi-class outputs
        fig = plt.figure(figsize=(8.5, 6.0))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[2], X_test_df, plot_type="bar",
                               class_names=CLASS_NAMES, show=False)
        else:
            shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {FINAL_MODEL_NAME} (PDAC class)", fontsize=12)
        plt.tight_layout()
        shap_summary_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(shap_summary_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print(f"          Saved → {shap_summary_path}")
    except Exception as e:
        print(f"          SHAP analysis skipped due to an error: {e}")
else:
    print("          shap not installed — skipping. Run `pip install shap` to enable this section.")

# =============================================================================
# SECTION 11 — ABLATION STUDY
# =============================================================================
print("\n[STEP 11]  Running feature-group ablation study …")

FEATURE_GROUPS = [
    ("Demographics only",                DEMOGRAPHIC_FEATURES),
    ("+ Lifestyle/history",              DEMOGRAPHIC_FEATURES + LIFESTYLE_FEATURES),
    ("+ Symptoms",                       DEMOGRAPHIC_FEATURES + LIFESTYLE_FEATURES + SYMPTOM_FEATURES),
    ("+ Biomarker (glucose)",            DEMOGRAPHIC_FEATURES + LIFESTYLE_FEATURES + SYMPTOM_FEATURES + BIOMARKER_FEATURES),
    ("+ Engineered scores (full model)", ALL_FEATURES),
]

ablation_rows = []
ablation_estimator = build_models()[BEST_MODEL_NAME]  # untuned architecture, isolates feature effect

for group_name, feats in FEATURE_GROUPS:
    Xg_train = pd.DataFrame(X_train, columns=ALL_FEATURES)[feats].values
    Xg_test  = pd.DataFrame(X_test,  columns=ALL_FEATURES)[feats].values
    pipe_g = make_pipeline(build_models()[BEST_MODEL_NAME])
    pipe_g.fit(Xg_train, y_train)
    pred_g = pipe_g.predict(Xg_test)
    ablation_rows.append({
        "Feature set":  group_name,
        "# Features":   len(feats),
        "Accuracy":     accuracy_score(y_test, pred_g),
        "F1 (macro)":   f1_score(y_test, pred_g, average="macro"),
    })
    print(f"          {group_name:<36} ({len(feats):>2} feats)  "
          f"Acc={ablation_rows[-1]['Accuracy']:.4f}  F1={ablation_rows[-1]['F1 (macro)']:.4f}")

ablation_df  = pd.DataFrame(ablation_rows)
ablation_csv = os.path.join(OUTPUT_DIR, "ablation_study.csv")
ablation_df.to_csv(ablation_csv, index=False)
print(f"          Ablation table saved → {ablation_csv}")

fig, ax = plt.subplots(figsize=(9.0, 5.0))
x_pos = np.arange(len(ablation_df))
ax.plot(x_pos, ablation_df["Accuracy"], marker="o", lw=2, color="#378ADD", label="Accuracy")
ax.plot(x_pos, ablation_df["F1 (macro)"], marker="s", lw=2, color="#1D9E75", label="F1 (macro)")
ax.set_xticks(x_pos)
ax.set_xticklabels(ablation_df["Feature set"], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Score", fontsize=10)
ax.set_title(f"Ablation Study — Incremental Feature-Group Contribution ({BEST_MODEL_NAME})", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
ablation_path = os.path.join(OUTPUT_DIR, "ablation_study.png")
plt.savefig(ablation_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print(f"          Saved → {ablation_path}")

# =============================================================================
# SECTION 12 — SAVE MODEL
# =============================================================================
print("\n[STEP 12]  Saving final tuned model …")

joblib.dump(best_model, MODEL_PATH)
print(f"          Model saved → {MODEL_PATH}")

artefacts = {
    "model"           : best_model,          # full Pipeline: scaler + tuned classifier
    "model_name"      : FINAL_MODEL_NAME,
    "cont_imputer"    : cont_imputer,
    "binary_imputer"  : binary_imputer,
    "base_features"   : BASE_FEATURES,
    "engineered_features": ENGINEERED_FEATURES,
    "feature_names"   : ALL_FEATURES,
    "class_names"     : CLASS_NAMES,
    "label_encoders"  : le_dict,
}
artefacts_path = os.path.join(OUTPUT_DIR, "pipeline_artefacts.pkl")
joblib.dump(artefacts, artefacts_path)
print(f"          Artefacts saved → {artefacts_path}")

# =============================================================================
# SECTION 13 — SAMPLE PREDICTIONS
# =============================================================================
print("\n[STEP 13]  Sample predictions …")

def predict_risk(patient: dict) -> dict:
    """
    Predict pancreatic cancer risk for a single patient.

    Parameters
    ----------
    patient : dict
        Keys must match BASE_FEATURES. Missing values as np.nan.
        Engineered features (symptom_score, lifestyle_risk_score) are
        computed automatically.

    Returns
    -------
    dict : predicted_class, risk_level, probabilities
    """
    row = pd.DataFrame([patient])[BASE_FEATURES]
    row[continuous_cols] = cont_imputer.transform(row[continuous_cols])
    row[binary_cols]     = binary_imputer.transform(row[binary_cols])

    row["symptom_score"] = (
        row["jaundice"] + row["weight_loss"] +
        row["abdominal_pain"] + row["back_pain"] + row["chronic_pancreatitis"]
    )
    row["lifestyle_risk_score"] = (
        row["smoking_history"] + row["diabetes_history"] + row["family_history"]
    )

    row = row[ALL_FEATURES]
    proba = best_model.predict_proba(row.values)[0]
    cls   = int(np.argmax(proba))
    return {
        "predicted_class" : CLASS_NAMES[cls],
        "risk_level"      : ["Low Risk", "Moderate Risk", "High Risk"][cls],
        "p_healthy"       : round(float(proba[0]), 4),
        "p_benign"        : round(float(proba[1]), 4),
        "p_pdac"          : round(float(proba[2]), 4),
        "confidence"      : f"{float(proba[cls]) * 100:.1f}%",
    }

patient_A = {  # High-risk profile
    "age": 67, "sex": 1, "glucose": 152.0, "smoking_history": 1,
    "diabetes_history": 1, "family_history": 1, "chronic_pancreatitis": 0,
    "jaundice": 1, "weight_loss": 1, "abdominal_pain": 1, "back_pain": 1,
}
patient_B = {  # Low-risk profile
    "age": 35, "sex": 0, "glucose": 90.0, "smoking_history": 0,
    "diabetes_history": 0, "family_history": 0, "chronic_pancreatitis": 0,
    "jaundice": 0, "weight_loss": 0, "abdominal_pain": 0, "back_pain": 0,
}
patient_C = {  # Wearable-only, some lab values missing
    "age": 58, "sex": 1, "glucose": 138.0, "smoking_history": 1,
    "diabetes_history": 1, "family_history": np.nan, "chronic_pancreatitis": 0,
    "jaundice": 0, "weight_loss": 1, "abdominal_pain": np.nan, "back_pain": 1,
}

for label, patient in [("A — High-risk  (67M)", patient_A),
                        ("B — Low-risk   (35F)", patient_B),
                        ("C — Wearable   (58M)", patient_C)]:
    result = predict_risk(patient)
    print(f"\n  Patient {label}")
    print(f"    Prediction  : {result['predicted_class']}  →  {result['risk_level']}")
    print(f"    Confidence  : {result['confidence']}")
    print(f"    P(Healthy)  : {result['p_healthy']:.4f}")
    print(f"    P(Benign)   : {result['p_benign']:.4f}")
    print(f"    P(PDAC)     : {result['p_pdac']:.4f}")

# =============================================================================
# SECTION 14 — SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  PIPELINE SUMMARY")
print("=" * 70)
print(f"  Dataset rows          : {len(df):,}")
print(f"  Base features         : {len(BASE_FEATURES)}")
print(f"  Engineered features   : {len(ENGINEERED_FEATURES)}")
print(f"  Total features used   : {len(ALL_FEATURES)}")
print(f"  Training samples      : {len(y_train):,}  (80%)")
print(f"  Testing  samples      : {len(y_test):,}  (20%)")
print(f"  Models compared       : {len(comparison_df)}")
print(f"  Best model (CV F1)    : {FINAL_MODEL_NAME}")
print(f"  Test  Accuracy        : {acc  * 100:.2f}%")
print(f"  Macro Precision       : {prec * 100:.2f}%")
print(f"  Macro Recall          : {rec  * 100:.2f}%")
print(f"  Macro F1-score        : {f1   * 100:.2f}%")
print(f"  ROC-AUC (macro)       : {roc_auc_macro:.4f}")
print(f"\n  Saved files:")
for p in [MODEL_PATH, artefacts_path, comparison_csv, model_cmp_path,
          dist_path, corr_path, cm_path, roc_path, pr_path, calib_path,
          fi_path, shap_summary_path, ablation_csv, ablation_path]:
    if p:
        print(f"    {p}")
print("=" * 70)
print("\n  [NOTE] Wearable features (real-time glucose, sweat pH,")
print("         sweat conductivity, heart rate, skin temperature)")
print("         will be fused at inference time via ESP32 data stream.")
print("         The saved model.pkl is ready for deployment integration.")
print("=" * 70 + "\n")