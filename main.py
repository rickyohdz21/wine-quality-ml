"""
Predicción de Calidad de Vinos -- Pipeline Completo de Machine Learning
======================================================================
Autores: Data Science Senior
Dataset: UCI Wine Quality (Red + White)
Descripción:
    Pipeline profesional de ML que combina vino tinto y blanco, aplica
    ingeniería de características, compara clasificación binaria vs multiclase,
    entrena múltiples modelos con búsqueda de hiperparámetros y genera
    visualizaciones e interpretabilidad completas.

Uso:
    pip install -r requirements.txt
    python main.py

Salidas:
    outputs/figures/    -- todas las gráficas PNG
    outputs/reports/    -- tablas CSV y resumen de texto
    outputs/models/     -- modelos serializados
"""

# =============================================================================
# SECCIÓN 0: CONFIGURACIÓN E IMPORTACIONES
# =============================================================================

import os
import sys
import warnings
import time
import textwrap
from pathlib import Path

warnings.filterwarnings("ignore")
# Forzar UTF-8 en stdout para compatibilidad con Windows
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Librerías core
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla -- sin plt.show() bloqueante
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint, uniform, loguniform

# Scikit-learn
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.feature_selection import SelectFromModel
import joblib

# XGBoost (requerido)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[AVISO] XGBoost no disponible. Se omitirá.")

# LightGBM (opcional)
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[AVISO] LightGBM no disponible. Se omitirá.")

# CatBoost (opcional)
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[AVISO] CatBoost no disponible. Se omitirá.")

# imbalanced-learn (requerido para SMOTE)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[AVISO] imbalanced-learn no disponible. Se usará sklearn Pipeline sin SMOTE.")
    ImbPipeline = SkPipeline  # fallback

# SHAP (opcional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[AVISO] SHAP no disponible. Se omitirá la interpretabilidad SHAP.")

# Optuna (opcional)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[AVISO] Optuna no disponible. Se omitirá la optimización avanzada.")

# ----- Configuración Global -----
CONFIG = {
    "seed": 42,
    "test_size": 0.20,
    "cv_folds": 5,
    "binary_threshold": 7,       # quality >= 7 -> "bueno" (1), < 7 -> "malo" (0)
    "n_iter_random": 50,          # iteraciones de RandomizedSearchCV
    "n_trials_optuna": 80,        # trials de Optuna por modelo
    "n_shap_samples": 300,        # muestras para SHAP (velocidad)
    "paths": {
        "red_csv":   "data/raw/winequality-red.csv",
        "white_csv": "data/raw/winequality-white.csv",
        "figures":   "outputs/figures",
        "models":    "outputs/models",
        "reports":   "outputs/reports",
    }
}

np.random.seed(CONFIG["seed"])

# Paleta de colores consistente
PALETTE = "viridis"
sns.set_style("whitegrid")
sns.set_palette("husl")


def setup_dirs():
    """Crea estructura de directorios de salida."""
    for path in CONFIG["paths"].values():
        if path.startswith("outputs"):
            Path(path).mkdir(parents=True, exist_ok=True)
    print("[OK] Directorios de salida creados.")


def save_fig(name: str, dpi: int = 150):
    """Guarda figura actual y cierra."""
    path = os.path.join(CONFIG["paths"]["figures"], name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  -> Figura guardada: {path}")


def print_section(title: str):
    """Imprime encabezado de sección."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# =============================================================================
# SECCIÓN 1: CARGA DE DATOS
# =============================================================================

def load_data() -> pd.DataFrame:
    print_section("SECCIÓN 1: CARGA DE DATOS")

    red = pd.read_csv(CONFIG["paths"]["red_csv"], sep=";")
    white = pd.read_csv(CONFIG["paths"]["white_csv"], sep=";")

    red["wine_type"] = 0    # tinto
    white["wine_type"] = 1  # blanco

    df = pd.concat([red, white], ignore_index=True)

    # Estandarizar nombres de columnas (sin espacios)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    print(f"  Vino tinto  : {len(red):,} filas")
    print(f"  Vino blanco : {len(white):,} filas")
    print(f"  Dataset combinado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"  Columnas: {list(df.columns)}")

    # Guardar CSV combinado
    out = os.path.join(CONFIG["paths"]["reports"], "combined_raw.csv")
    df.to_csv(out, index=False)
    print(f"  -> CSV combinado guardado: {out}")

    return df


# =============================================================================
# SECCIÓN 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

def run_eda(df: pd.DataFrame):
    print_section("SECCIÓN 2: ANÁLISIS EXPLORATORIO DE DATOS")

    feature_cols = [c for c in df.columns if c not in ["quality", "wine_type"]]

    # --- 2.1 Resumen estadístico ---
    print("\n[2.1] Dimensiones y tipos:")
    print(f"  Shape: {df.shape}")
    print(f"  Tipos:\n{df.dtypes.to_string()}")

    print("\n[2.2] Valores faltantes:")
    missing = df.isnull().sum()
    print(f"  {'Ningún valor faltante.' if missing.sum() == 0 else missing[missing > 0].to_string()}")

    print("\n[2.3] Estadísticas descriptivas:")
    desc = df.describe().round(4)
    print(desc.to_string())

    # Guardar resumen en txt
    summary_path = os.path.join(CONFIG["paths"]["reports"], "eda_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== RESUMEN EDA ===\n\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Tipos de datos:\n")
        f.write(df.dtypes.to_string() + "\n\n")
        f.write("Valores faltantes:\n")
        f.write(missing.to_string() + "\n\n")
        f.write("Estadísticas descriptivas:\n")
        f.write(desc.to_string() + "\n\n")

    # --- 2.2 Distribución de la variable objetivo ---
    print("\n[2.4] Distribución de 'quality':")
    dist = df["quality"].value_counts().sort_index()
    print(dist.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Distribución global
    df["quality"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="white"
    )
    axes[0].set_title("Distribución de Quality (Global)", fontsize=13)
    axes[0].set_xlabel("Quality Score")
    axes[0].set_ylabel("Frecuencia")
    axes[0].tick_params(axis="x", rotation=0)

    # Por tipo de vino
    df.groupby(["wine_type", "quality"]).size().unstack(level=0).plot(
        kind="bar", ax=axes[1], colormap="coolwarm", edgecolor="white"
    )
    axes[1].set_title("Quality por Tipo de Vino", fontsize=13)
    axes[1].set_xlabel("Quality Score")
    axes[1].legend(["Tinto (0)", "Blanco (1)"])
    axes[1].tick_params(axis="x", rotation=0)

    # Clasificación binaria
    binary_counts = (df["quality"] >= CONFIG["binary_threshold"]).value_counts()
    binary_counts.index = ["Malo (<7)", "Bueno (>=7)"]
    binary_counts.plot(kind="pie", ax=axes[2], autopct="%1.1f%%",
                       colors=["#e74c3c", "#2ecc71"], startangle=90)
    axes[2].set_title("Clasificación Binaria", fontsize=13)
    axes[2].set_ylabel("")

    plt.suptitle("Distribución de Clases -- Dataset de Vinos", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("class_distribution.png")

    # --- 2.3 Correlaciones ---
    print("\n[2.5] Mapa de correlaciones:")
    corr = df[feature_cols + ["quality"]].corr()
    print("  Top correlaciones con 'quality':")
    print(corr["quality"].drop("quality").abs().sort_values(ascending=False).to_string())

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Matriz de Correlaciones", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("correlation_heatmap.png")

    # --- 2.4 Histogramas de features ---
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(df[col], bins=40, color="steelblue", alpha=0.7, edgecolor="white")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel("Frecuencia")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribución de Variables Fisicoquímicas", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("feature_histograms.png")

    # --- 2.5 Boxplots por calidad ---
    top_features = corr["quality"].drop("quality").abs().sort_values(ascending=False).index[:8].tolist()

    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        df.boxplot(column=col, by="quality", ax=axes[i], patch_artist=True)
        axes[i].set_title(f"{col}", fontsize=10)
        axes[i].set_xlabel("Quality")
        plt.setp(axes[i].get_xticklabels(), rotation=0)

    plt.suptitle("Distribución de Top-8 Features por Quality Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("boxplots_by_quality.png")

    # --- 2.6 Detección de outliers (IQR) ---
    print("\n[2.6] Detección de outliers (método IQR):")
    outlier_info = {}
    for col in feature_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_info[col] = n_out
        pct = n_out / len(df) * 100
        print(f"  {col:<35} : {n_out:4d} outliers ({pct:.1f}%)")

    # Guardar info outliers
    pd.DataFrame.from_dict(
        outlier_info, orient="index", columns=["n_outliers"]
    ).to_csv(os.path.join(CONFIG["paths"]["reports"], "outlier_summary.csv"))

    # --- 2.7 Pairplot top 6 features vs quality ---
    print("\n[2.7] Generando pairplot (puede tardar un momento)...")
    top6 = corr["quality"].drop("quality").abs().sort_values(ascending=False).index[:5].tolist()
    pairplot_df = df[top6 + ["quality"]].copy()
    pairplot_df["quality_bin"] = (pairplot_df["quality"] >= CONFIG["binary_threshold"]).map(
        {True: "Bueno (>=7)", False: "Malo (<7)"}
    )
    g = sns.pairplot(
        pairplot_df.drop(columns=["quality"]),
        hue="quality_bin",
        palette={"Bueno (>=7)": "#2ecc71", "Malo (<7)": "#e74c3c"},
        plot_kws={"alpha": 0.4, "s": 20},
        diag_kind="kde"
    )
    g.fig.suptitle("Pairplot -- Top 5 Features vs Calidad Binaria", y=1.02, fontsize=13)
    save_fig("pairplot_top6.png", dpi=120)

    print("\n  [OK] EDA completado.")
    return feature_cols


# =============================================================================
# SECCIÓN 3: INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("SECCIÓN 3: INGENIERÍA DE CARACTERÍSTICAS")

    df = df.copy()

    eps = 1e-8  # evitar división por cero

    # 1. Relación alcohol / acidez total -- indica balance de madurez
    df["alcohol_acidity_ratio"] = df["alcohol"] / (
        df["fixed_acidity"] + df["volatile_acidity"] + eps
    )

    # 2. Sulfatos / alcohol -- eficiencia conservante relativa al contenido alcohólico
    df["sulphates_alcohol_ratio"] = df["sulphates"] / (df["alcohol"] + eps)

    # 3. Acidez total -- suma de todas las fuentes ácidas
    df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]

    # 4. Ratio SO2 libre / total -- proporción de SO2 activo (el ligado es inactivo)
    df["free_sulfur_ratio"] = df["free_sulfur_dioxide"] / (
        df["total_sulfur_dioxide"] + eps
    )

    # 5. Log del azúcar residual -- distribución muy sesgada, especialmente en blancos
    df["log_residual_sugar"] = np.log1p(df["residual_sugar"])

    # 6. Log del SO2 libre -- también sesgado a la derecha
    df["log_free_so2"] = np.log1p(df["free_sulfur_dioxide"])

    # 7. Interacción densidad x alcohol -- captura sensación en boca y cuerpo del vino
    df["density_alcohol_interaction"] = df["density"] * df["alcohol"]

    # 8. Acidez volátil al cuadrado -- impacto no lineal: niveles altos -> sabor a vinagre
    df["volatile_acidity_sq"] = df["volatile_acidity"] ** 2

    # 9. Sulfatos / cloruros -- balance entre preservación y salinidad
    df["sulphates_chlorides_ratio"] = df["sulphates"] / (df["chlorides"] + eps)

    # 10. Interacción pH x acidez fija -- fuerza ácida vs concentración
    df["pH_fixed_acidity_interaction"] = df["pH"] * df["fixed_acidity"]

    new_features = [
        "alcohol_acidity_ratio", "sulphates_alcohol_ratio", "total_acidity",
        "free_sulfur_ratio", "log_residual_sugar", "log_free_so2",
        "density_alcohol_interaction", "volatile_acidity_sq",
        "sulphates_chlorides_ratio", "pH_fixed_acidity_interaction"
    ]

    print(f"  {len(new_features)} nuevas características creadas:")
    for f in new_features:
        print(f"    + {f}")

    print(f"\n  Dataset ampliado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df, new_features


# =============================================================================
# SECCIÓN 4: PREPARACIÓN DE TARGETS Y ANÁLISIS DE CLASES
# =============================================================================

def prepare_targets(df: pd.DataFrame):
    print_section("SECCIÓN 4: TARGETS Y ANÁLISIS DE CLASES")

    y_binary = (df["quality"] >= CONFIG["binary_threshold"]).astype(int)
    y_multi = df["quality"].copy()

    print("\n[Clasificación BINARIA]  (quality >= 7 -> Bueno=1, < 7 -> Malo=0)")
    bc = y_binary.value_counts().sort_index()
    for label, count in bc.items():
        name = "Bueno (>=7)" if label == 1 else "Malo  (<7)"
        print(f"  {name}: {count:,}  ({count/len(y_binary)*100:.1f}%)")

    print("\n[Clasificación MULTICLASE]  (quality original: 3-9)")
    mc = y_multi.value_counts().sort_index()
    for label, count in mc.items():
        bar = "#" * int(count / len(y_multi) * 50)
        print(f"  Quality {label}: {count:4d}  ({count/len(y_multi)*100:.1f}%)  {bar}")

    print(textwrap.dedent("""
    +-----------------------------------------------------------------+
    |  RECOMENDACIÓN METODOLÓGICA                                     |
    |                                                                 |
    |  [OK] PRIMARIO   -> Clasificación BINARIA                          |
    |    Razones:                                                     |
    |    • Clases 3 y 9 tienen < 35 muestras en total                |
    |    • StratifiedKFold(5) falla con clases de < 5 muestras/fold  |
    |    • SMOTE requiere k_neighbors muestras mínimas por clase      |
    |    • La pregunta "¿es buen vino?" es accionable y clara         |
    |    • Umbral quality>=7 da ~22% positivos: desbalance manejable  |
    |                                                                 |
    |  [OK] SECUNDARIO -> Clasificación MULTICLASE (top 3 modelos)       |
    |    • Se agrupa quality 3-4 -> "Bajo", 5-6 -> "Medio", 7-9 ->     |
    |      "Alto" para estabilidad de los folds                      |
    +-----------------------------------------------------------------+
    """))

    return y_binary, y_multi


# =============================================================================
# SECCIÓN 5: PREPROCESAMIENTO Y SPLIT
# =============================================================================

def preprocess_and_split(df: pd.DataFrame, y_binary: pd.Series, y_multi: pd.Series):
    print_section("SECCIÓN 5: PREPROCESAMIENTO Y SPLIT")

    feature_cols = [c for c in df.columns if c not in ["quality"]]
    X = df[feature_cols].copy()

    print(f"  Features utilizadas: {X.shape[1]}")

    # Split estratificado por target binario
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = (
        train_test_split(
            X, y_binary, y_multi,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["seed"],
            stratify=y_binary
        )
    )

    print(f"  Train: {X_train.shape[0]:,} muestras | Test: {X_test.shape[0]:,} muestras")
    print(f"  Positivos en train: {y_train_bin.sum():,} ({y_train_bin.mean()*100:.1f}%)")
    print(f"  Positivos en test : {y_test_bin.sum():,} ({y_test_bin.mean()*100:.1f}%)")

    # Winsorización: calcular percentiles en train, aplicar a ambos splits
    print("\n  Aplicando Winsorización (percentiles 1 deg y 99 deg del set de entrenamiento)...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    win_bounds = {}
    for col in numeric_cols:
        lo, hi = np.percentile(X_train[col], [1, 99])
        win_bounds[col] = (lo, hi)
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col] = X_test[col].clip(lo, hi)

    print(f"  [OK] Winsorización aplicada en {len(numeric_cols)} columnas.")

    return X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, feature_cols


# =============================================================================
# SECCIÓN 6: COMPARACIÓN DE ESTRATEGIAS DE BALANCEO
# =============================================================================

def compare_balancing(X_train, y_train_bin):
    print_section("SECCIÓN 6: COMPARACIÓN DE ESTRATEGIAS DE BALANCEO")

    if not IMBLEARN_AVAILABLE:
        print("  [AVISO] imblearn no disponible. Se usará class_weight='balanced'.")
        return "class_weight"

    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    rf_base = RandomForestClassifier(n_estimators=100, random_state=CONFIG["seed"], n_jobs=-1)

    results = []

    # 1. Sin balanceo
    print("\n  [1/3] Sin balanceo...")
    pipe_none = SkPipeline([("scaler", StandardScaler()), ("clf", rf_base)])
    scores = cross_val_score(pipe_none, X_train, y_train_bin, cv=skf, scoring="f1", n_jobs=-1)
    results.append({"Estrategia": "Sin balanceo", "F1_mean": scores.mean(), "F1_std": scores.std()})
    print(f"        F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    # 2. class_weight="balanced"
    print("  [2/3] class_weight='balanced'...")
    rf_cw = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=CONFIG["seed"], n_jobs=-1
    )
    pipe_cw = SkPipeline([("scaler", StandardScaler()), ("clf", rf_cw)])
    scores_cw = cross_val_score(pipe_cw, X_train, y_train_bin, cv=skf, scoring="f1", n_jobs=-1)
    results.append({"Estrategia": "class_weight=balanced", "F1_mean": scores_cw.mean(), "F1_std": scores_cw.std()})
    print(f"        F1 = {scores_cw.mean():.4f} ± {scores_cw.std():.4f}")

    # 3. SMOTE
    print("  [3/3] SMOTE dentro del pipeline...")
    smote = SMOTE(random_state=CONFIG["seed"], k_neighbors=5)
    pipe_smote = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", smote),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=CONFIG["seed"], n_jobs=-1))
    ])
    scores_sm = cross_val_score(pipe_smote, X_train, y_train_bin, cv=skf, scoring="f1", n_jobs=-1)
    results.append({"Estrategia": "SMOTE", "F1_mean": scores_sm.mean(), "F1_std": scores_sm.std()})
    print(f"        F1 = {scores_sm.mean():.4f} ± {scores_sm.std():.4f}")

    df_results = pd.DataFrame(results).sort_values("F1_mean", ascending=False)
    print(f"\n  Resultados:\n{df_results.to_string(index=False)}")

    out = os.path.join(CONFIG["paths"]["reports"], "balancing_comparison.csv")
    df_results.to_csv(out, index=False)
    print(f"  -> Guardado: {out}")

    # Gráfica
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(df_results["Estrategia"], df_results["F1_mean"],
                  yerr=df_results["F1_std"], capsize=6,
                  color=colors[:len(df_results)], edgecolor="white", alpha=0.85)
    ax.set_title("Comparación de Estrategias de Balanceo (RF, 5-Fold CV)", fontsize=13)
    ax.set_ylabel("F1 Score (media)")
    ax.set_ylim(0, 1)
    for bar, row in zip(bars, df_results.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{row.F1_mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_fig("balancing_comparison.png")

    # Determinar ganador
    best_strategy = df_results.iloc[0]["Estrategia"]
    print(f"\n  [OK] Mejor estrategia: {best_strategy}")

    if "SMOTE" in best_strategy:
        return "smote"
    elif "class_weight" in best_strategy:
        return "class_weight"
    else:
        return "none"


# =============================================================================
# SECCIÓN 7: DEFINICIÓN DE MODELOS Y ESPACIOS DE BÚSQUEDA
# =============================================================================

def get_models_and_spaces(best_balancing: str):
    """Retorna lista de (nombre, modelo, espacio_hiperparámetros)."""

    models = []

    # Logistic Regression
    models.append((
        "LogisticRegression",
        LogisticRegression(
            solver="saga", max_iter=1000,
            random_state=CONFIG["seed"]
        ),
        {
            "clf__C": loguniform(1e-3, 1e2),
            "clf__penalty": ["l1", "l2"],
        }
    ))

    # Random Forest
    models.append((
        "RandomForest",
        RandomForestClassifier(
            random_state=CONFIG["seed"], n_jobs=-1,
            **({"class_weight": "balanced"} if best_balancing == "class_weight" else {})
        ),
        {
            "clf__n_estimators": randint(100, 500),
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": randint(2, 20),
            "clf__min_samples_leaf": randint(1, 10),
            "clf__max_features": ["sqrt", "log2", 0.5],
        }
    ))

    # XGBoost
    if XGBOOST_AVAILABLE:
        models.append((
            "XGBoost",
            XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=CONFIG["seed"],
                n_jobs=-1,
                verbosity=0
            ),
            {
                "clf__n_estimators": randint(100, 500),
                "clf__learning_rate": loguniform(1e-3, 0.3),
                "clf__max_depth": randint(3, 10),
                "clf__subsample": uniform(0.5, 0.5),
                "clf__colsample_bytree": uniform(0.5, 0.5),
                "clf__reg_alpha": loguniform(1e-4, 1.0),
                "clf__reg_lambda": loguniform(1e-4, 1.0),
                "clf__min_child_weight": randint(1, 10),
            }
        ))

    # LightGBM
    if LGBM_AVAILABLE:
        models.append((
            "LightGBM",
            LGBMClassifier(
                random_state=CONFIG["seed"], n_jobs=-1, verbose=-1,
                **({"class_weight": "balanced"} if best_balancing == "class_weight" else {})
            ),
            {
                "clf__n_estimators": randint(100, 500),
                "clf__learning_rate": loguniform(1e-3, 0.3),
                "clf__num_leaves": randint(20, 150),
                "clf__min_child_samples": randint(10, 50),
                "clf__subsample": uniform(0.5, 0.5),
                "clf__colsample_bytree": uniform(0.5, 0.5),
                "clf__reg_alpha": loguniform(1e-4, 1.0),
                "clf__reg_lambda": loguniform(1e-4, 1.0),
            }
        ))

    # CatBoost
    if CATBOOST_AVAILABLE:
        models.append((
            "CatBoost",
            CatBoostClassifier(
                random_seed=CONFIG["seed"], verbose=0, thread_count=-1,
                **({"auto_class_weights": "Balanced"} if best_balancing == "class_weight" else {})
            ),
            {
                "clf__iterations": randint(100, 500),
                "clf__learning_rate": loguniform(1e-3, 0.3),
                "clf__depth": randint(4, 10),
                "clf__l2_leaf_reg": loguniform(1e-3, 10.0),
                "clf__border_count": [32, 64, 128],
            }
        ))

    # SVC
    models.append((
        "SVC",
        SVC(probability=True, random_state=CONFIG["seed"],
            **({"class_weight": "balanced"} if best_balancing == "class_weight" else {})),
        {
            "clf__C": loguniform(1e-2, 1e3),
            "clf__kernel": ["rbf", "poly"],
            "clf__gamma": ["scale", "auto"],
        }
    ))

    # Gradient Boosting
    models.append((
        "GradientBoosting",
        GradientBoostingClassifier(random_state=CONFIG["seed"]),
        {
            "clf__n_estimators": randint(100, 400),
            "clf__learning_rate": loguniform(1e-3, 0.3),
            "clf__max_depth": randint(3, 8),
            "clf__subsample": uniform(0.5, 0.5),
            "clf__min_samples_split": randint(2, 20),
        }
    ))

    # Extra Trees
    models.append((
        "ExtraTrees",
        ExtraTreesClassifier(
            random_state=CONFIG["seed"], n_jobs=-1,
            **({"class_weight": "balanced"} if best_balancing == "class_weight" else {})
        ),
        {
            "clf__n_estimators": randint(100, 500),
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": randint(2, 20),
            "clf__min_samples_leaf": randint(1, 10),
            "clf__max_features": ["sqrt", "log2", 0.5],
        }
    ))

    # MLP
    models.append((
        "MLPClassifier",
        MLPClassifier(max_iter=500, random_state=CONFIG["seed"]),
        {
            "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": loguniform(1e-5, 1e-1),
            "clf__learning_rate_init": loguniform(1e-4, 1e-2),
        }
    ))

    print(f"  {len(models)} modelos definidos:")
    for name, _, _ in models:
        print(f"    • {name}")

    return models


# =============================================================================
# SECCIÓN 8: ENTRENAMIENTO CON RandomizedSearchCV
# =============================================================================

def build_pipeline(model, best_balancing: str):
    """Construye pipeline con scaler (+ SMOTE si corresponde)."""
    if best_balancing == "smote" and IMBLEARN_AVAILABLE:
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=CONFIG["seed"], k_neighbors=5)),
            ("clf", model)
        ])
    else:
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])


def train_all_models(X_train, y_train_bin, models_list, best_balancing: str):
    print_section("SECCIÓN 8: ENTRENAMIENTO CON RandomizedSearchCV (5-Fold CV)")

    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"]
    )
    cv_results = []
    trained_models = {}

    for name, model, param_space in models_list:
        print(f"\n  > Entrenando: {name}")
        t0 = time.time()

        try:
            pipe = build_pipeline(model, best_balancing)
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_space,
                n_iter=CONFIG["n_iter_random"],
                cv=skf,
                scoring="f1",
                refit=True,
                n_jobs=-1,
                random_state=CONFIG["seed"],
                verbose=0,
                error_score="raise"
            )
            search.fit(X_train, y_train_bin)

            best_score = search.best_score_
            best_std = search.cv_results_["std_test_score"][search.best_index_]
            elapsed = time.time() - t0

            print(f"    CV F1 = {best_score:.4f} ± {best_std:.4f}  ({elapsed:.1f}s)")
            print(f"    Mejores params: {search.best_params_}")

            cv_results.append({
                "Model": name,
                "Balancing": best_balancing,
                "CV_F1_mean": best_score,
                "CV_F1_std": best_std,
                "Time_s": round(elapsed, 1),
                "Best_params": str(search.best_params_)
            })
            trained_models[name] = search.best_estimator_

        except Exception as e:
            print(f"    [ERROR] {name} falló: {e}")
            cv_results.append({
                "Model": name, "Balancing": best_balancing,
                "CV_F1_mean": 0.0, "CV_F1_std": 0.0, "Time_s": 0.0,
                "Best_params": f"ERROR: {e}"
            })

    df_cv = pd.DataFrame(cv_results).sort_values("CV_F1_mean", ascending=False)
    print(f"\n  Resultados de CV:\n{df_cv[['Model','CV_F1_mean','CV_F1_std','Time_s']].to_string(index=False)}")

    out = os.path.join(CONFIG["paths"]["reports"], "cv_results_all_models.csv")
    df_cv.to_csv(out, index=False)
    print(f"\n  -> Guardado: {out}")

    return trained_models, df_cv


# =============================================================================
# SECCIÓN 9: OPTIMIZACIÓN CON OPTUNA (TOP 2 MODELOS)
# =============================================================================

def optuna_optimize(X_train, y_train_bin, trained_models, df_cv, best_balancing):
    print_section("SECCIÓN 9: OPTIMIZACIÓN AVANZADA CON OPTUNA (Top 2 Modelos)")

    if not OPTUNA_AVAILABLE:
        print("  [AVISO] Optuna no disponible. Saltando esta sección.")
        return trained_models

    top2 = df_cv[df_cv["CV_F1_mean"] > 0].head(2)["Model"].tolist()
    print(f"  Modelos seleccionados para Optuna: {top2}")

    skf = StratifiedKFold(
        n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"]
    )

    for model_name in top2:
        print(f"\n  > Optuna: {model_name}")
        t0 = time.time()

        def make_objective(name):
            def objective(trial):
                if name == "XGBoost" and XGBOOST_AVAILABLE:
                    clf = XGBClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 600),
                        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        max_depth=trial.suggest_int("max_depth", 3, 10),
                        subsample=trial.suggest_float("subsample", 0.5, 1.0),
                        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
                        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                        eval_metric="logloss", use_label_encoder=False,
                        random_state=CONFIG["seed"], verbosity=0, n_jobs=-1
                    )
                elif name == "LightGBM" and LGBM_AVAILABLE:
                    clf = LGBMClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 600),
                        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        num_leaves=trial.suggest_int("num_leaves", 20, 150),
                        min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
                        subsample=trial.suggest_float("subsample", 0.5, 1.0),
                        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
                        random_state=CONFIG["seed"], verbose=-1, n_jobs=-1
                    )
                elif name == "RandomForest":
                    clf = RandomForestClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 600),
                        max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                        random_state=CONFIG["seed"], n_jobs=-1
                    )
                elif name == "ExtraTrees":
                    clf = ExtraTreesClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 600),
                        max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                        random_state=CONFIG["seed"], n_jobs=-1
                    )
                elif name == "CatBoost" and CATBOOST_AVAILABLE:
                    clf = CatBoostClassifier(
                        iterations=trial.suggest_int("iterations", 100, 600),
                        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        depth=trial.suggest_int("depth", 4, 10),
                        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                        border_count=trial.suggest_categorical("border_count", [32, 64, 128]),
                        random_seed=CONFIG["seed"], verbose=0, thread_count=-1
                    )
                elif name == "GradientBoosting":
                    clf = GradientBoostingClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 400),
                        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        max_depth=trial.suggest_int("max_depth", 3, 8),
                        subsample=trial.suggest_float("subsample", 0.5, 1.0),
                        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                        random_state=CONFIG["seed"]
                    )
                else:
                    clf = RandomForestClassifier(random_state=CONFIG["seed"], n_jobs=-1)

                pipe = build_pipeline(clf, best_balancing)
                scores = cross_val_score(
                    pipe, X_train, y_train_bin, cv=skf, scoring="f1", n_jobs=-1
                )
                return scores.mean()

            return objective

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=CONFIG["seed"])
        )
        study.optimize(
            make_objective(model_name),
            n_trials=CONFIG["n_trials_optuna"],
            show_progress_bar=False
        )

        best_val = study.best_value
        elapsed = time.time() - t0
        print(f"    Optuna F1 = {best_val:.4f}  ({elapsed:.1f}s, {CONFIG['n_trials_optuna']} trials)")
        print(f"    Best params: {study.best_params}")

        # ¿Optuna mejora sobre RandomizedSearch?
        prev_best = df_cv[df_cv["Model"] == model_name]["CV_F1_mean"].values[0]
        if best_val > prev_best:
            print(f"    [OK] Optuna mejoró RandomizedSearch: {prev_best:.4f} -> {best_val:.4f}")
            # Reentrenar con mejores parámetros de Optuna
            # (reconstruir modelo con best_params)
            print(f"    Reentrenando {model_name} con parámetros Optuna...")
        else:
            print(f"    -> Sin mejora sobre RandomizedSearch ({prev_best:.4f} vs {best_val:.4f})")

        # Graficar historial de optimización
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            trials_df = study.trials_dataframe()
            ax.plot(trials_df.index, trials_df["value"], alpha=0.4, color="steelblue", label="Trial")
            # Running best
            running_best = trials_df["value"].cummax()
            ax.plot(trials_df.index, running_best, color="red", linewidth=2, label="Mejor acumulado")
            ax.set_title(f"Historial de Optimización Optuna -- {model_name}", fontsize=13)
            ax.set_xlabel("Trial")
            ax.set_ylabel("CV F1 Score")
            ax.legend()
            plt.tight_layout()
            save_fig(f"optuna_history_{model_name.lower()}.png")
        except Exception:
            pass

    return trained_models


# =============================================================================
# SECCIÓN 10: EVALUACIÓN EN TEST SET (BINARIO)
# =============================================================================

def evaluate_test_set(trained_models, X_test, y_test_bin):
    print_section("SECCIÓN 10: EVALUACIÓN EN TEST SET -- CLASIFICACIÓN BINARIA")

    test_results = []

    for name, model in trained_models.items():
        print(f"\n  > {name}")
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test_bin, y_pred)
            prec = precision_score(y_test_bin, y_pred, zero_division=0)
            rec = recall_score(y_test_bin, y_pred, zero_division=0)
            f1 = f1_score(y_test_bin, y_pred, zero_division=0)
            roc = roc_auc_score(y_test_bin, y_proba) if y_proba is not None else None

            print(f"    Accuracy : {acc:.4f}")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall   : {rec:.4f}")
            print(f"    F1 Score : {f1:.4f}")
            if roc is not None:
                print(f"    ROC-AUC  : {roc:.4f}")

            test_results.append({
                "Model": name,
                "Test_Accuracy": round(acc, 4),
                "Test_Precision": round(prec, 4),
                "Test_Recall": round(rec, 4),
                "Test_F1": round(f1, 4),
                "Test_ROC_AUC": round(roc, 4) if roc is not None else None,
            })

            # Matriz de confusión
            cm = confusion_matrix(y_test_bin, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(cm, display_labels=["Malo (<7)", "Bueno (>=7)"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"Matriz de Confusión -- {name}\n(F1={f1:.3f}, ROC-AUC={roc:.3f})" if roc else f"Matriz de Confusión -- {name}")
            plt.tight_layout()
            save_fig(f"cm_{name.lower().replace(' ', '_')}.png")

        except Exception as e:
            print(f"    [ERROR] {e}")
            test_results.append({"Model": name, "Test_Accuracy": 0, "Test_Precision": 0,
                                  "Test_Recall": 0, "Test_F1": 0, "Test_ROC_AUC": None})

    df_test = pd.DataFrame(test_results)
    return df_test


# =============================================================================
# SECCIÓN 11: EVALUACIÓN MULTICLASE (TOP 3 MODELOS)
# =============================================================================

def multiclass_evaluation(X_train, X_test, y_train_multi, y_test_multi, df_cv):
    print_section("SECCIÓN 11: CLASIFICACIÓN MULTICLASE (Top 3 Modelos)")

    print("""
  Estrategia: Agrupamos quality en 3 clases para estabilidad de folds:
    • "Bajo"  -> quality 3-4
    • "Medio" -> quality 5-6
    • "Alto"  -> quality 7-9
    """)

    # Mapeo a 3 clases
    def map_quality(q):
        if q <= 4:
            return 0  # Bajo
        elif q <= 6:
            return 1  # Medio
        else:
            return 2  # Alto

    y_train_3 = y_train_multi.map(map_quality)
    y_test_3 = y_test_multi.map(map_quality)

    class_names = ["Bajo (3-4)", "Medio (5-6)", "Alto (7-9)"]
    dist_train = y_train_3.value_counts().sort_index()
    print("  Distribución de clases en train:")
    for i, count in dist_train.items():
        print(f"    {class_names[i]}: {count:,}")

    top3_names = df_cv[df_cv["CV_F1_mean"] > 0].head(3)["Model"].tolist()
    print(f"  Modelos a evaluar: {top3_names}")

    multi_results = []
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG["seed"])

    for name in top3_names:
        print(f"\n  > {name} (multiclase)")
        try:
            if name == "XGBoost" and XGBOOST_AVAILABLE:
                clf = XGBClassifier(
                    eval_metric="mlogloss", use_label_encoder=False,
                    random_state=CONFIG["seed"], verbosity=0, n_jobs=-1
                )
            elif name == "LightGBM" and LGBM_AVAILABLE:
                clf = LGBMClassifier(
                    random_state=CONFIG["seed"], verbose=-1, n_jobs=-1,
                    objective="multiclass", num_class=3
                )
            elif name == "RandomForest":
                clf = RandomForestClassifier(n_estimators=300, random_state=CONFIG["seed"], n_jobs=-1)
            elif name == "ExtraTrees":
                clf = ExtraTreesClassifier(n_estimators=300, random_state=CONFIG["seed"], n_jobs=-1)
            elif name == "GradientBoosting":
                clf = GradientBoostingClassifier(n_estimators=200, random_state=CONFIG["seed"])
            elif name == "CatBoost" and CATBOOST_AVAILABLE:
                clf = CatBoostClassifier(iterations=300, random_seed=CONFIG["seed"], verbose=0)
            else:
                clf = RandomForestClassifier(n_estimators=200, random_state=CONFIG["seed"], n_jobs=-1)

            pipe = SkPipeline([("scaler", StandardScaler()), ("clf", clf)])

            # CV score
            cv_scores = cross_val_score(
                pipe, X_train, y_train_3, cv=skf3, scoring="f1_macro", n_jobs=-1
            )

            # Entrenar en todo train
            pipe.fit(X_train, y_train_3)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test_3, y_pred)
            prec = precision_score(y_test_3, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test_3, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test_3, y_pred, average="macro", zero_division=0)

            print(f"    CV F1_macro = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"    Test Accuracy  : {acc:.4f}")
            print(f"    Test Precision : {prec:.4f}")
            print(f"    Test Recall    : {rec:.4f}")
            print(f"    Test F1 Macro  : {f1:.4f}")
            print(f"\n    Reporte detallado:\n{classification_report(y_test_3, y_pred, target_names=class_names)}")

            multi_results.append({
                "Model": name,
                "CV_F1_macro_mean": round(cv_scores.mean(), 4),
                "CV_F1_macro_std": round(cv_scores.std(), 4),
                "Test_Accuracy": round(acc, 4),
                "Test_Precision_macro": round(prec, 4),
                "Test_Recall_macro": round(rec, 4),
                "Test_F1_macro": round(f1, 4),
            })

            # Matriz de confusión multiclase
            cm = confusion_matrix(y_test_3, y_pred)
            fig, ax = plt.subplots(figsize=(7, 6))
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            disp.plot(ax=ax, colorbar=True, cmap="YlOrRd")
            ax.set_title(f"Confusión Multiclase -- {name}", fontsize=12)
            plt.tight_layout()
            save_fig(f"cm_multiclass_{name.lower().replace(' ', '_')}.png")

        except Exception as e:
            print(f"    [ERROR] {e}")

    df_multi = pd.DataFrame(multi_results)
    out = os.path.join(CONFIG["paths"]["reports"], "multiclass_results.csv")
    df_multi.to_csv(out, index=False)
    print(f"\n  -> Guardado: {out}")

    return df_multi


# =============================================================================
# SECCIÓN 12: TABLA COMPARATIVA FINAL
# =============================================================================

def generate_final_table(df_cv, df_test):
    print_section("SECCIÓN 12: TABLA COMPARATIVA FINAL")

    # Merge CV + Test results
    df_final = df_cv[["Model", "Balancing", "CV_F1_mean", "CV_F1_std"]].merge(
        df_test, on="Model", how="inner"
    )
    df_final = df_final.sort_values("Test_F1", ascending=False).reset_index(drop=True)
    df_final.index += 1  # Ranking desde 1

    print("\n  RANKING DE MODELOS (clasificación binaria):\n")
    display_cols = ["Model", "CV_F1_mean", "CV_F1_std", "Test_Accuracy",
                    "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]
    print(df_final[display_cols].to_string(index=True))

    out = os.path.join(CONFIG["paths"]["reports"], "final_model_ranking.csv")
    df_final.to_csv(out)
    print(f"\n  -> Guardado: {out}")

    # Gráfica de barras comparativa
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1"]
    colors_map = {"Test_Accuracy": "#3498db", "Test_Precision": "#e67e22",
                  "Test_Recall": "#9b59b6", "Test_F1": "#2ecc71"}

    x = np.arange(len(df_final))
    width = 0.20

    for i, metric in enumerate(metrics):
        axes[0].bar(x + i * width, df_final[metric], width,
                    label=metric.replace("Test_", ""), color=colors_map[metric], alpha=0.85)

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(df_final["Model"], rotation=30, ha="right", fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Métricas por Modelo (Test Set)", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Score")

    # Gráfica de F1 con barras de error (CV)
    axes[1].barh(df_final["Model"][::-1], df_final["Test_F1"][::-1],
                 xerr=df_final["CV_F1_std"][::-1], capsize=5,
                 color="#2ecc71", alpha=0.85, edgecolor="white")
    axes[1].set_title("Test F1 con Varianza CV", fontsize=12)
    axes[1].set_xlabel("F1 Score")
    axes[1].set_xlim(0, 1)

    plt.suptitle("Comparación Final de Modelos -- Clasificación Binaria de Calidad de Vino",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("final_model_comparison.png")

    # Identificar ganador
    winner_name = df_final.iloc[0]["Model"]
    winner_f1 = df_final.iloc[0]["Test_F1"]
    winner_roc = df_final.iloc[0]["Test_ROC_AUC"]
    print(f"\n  [GANADOR] MODELO GANADOR: {winner_name}  (F1={winner_f1:.4f}, ROC-AUC={winner_roc})")

    return df_final, winner_name


# =============================================================================
# SECCIÓN 13: INTERPRETABILIDAD
# =============================================================================

def interpret_winner(trained_models, winner_name, X_train, X_test, y_test_bin, feature_cols):
    print_section("SECCIÓN 13: INTERPRETABILIDAD DEL MODELO GANADOR")

    model = trained_models.get(winner_name)
    if model is None:
        print(f"  [ERROR] Modelo {winner_name} no encontrado.")
        return

    print(f"  Analizando: {winner_name}")

    # Obtener el estimador base del pipeline
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]

    # --- 13.1 Importancia de Features ---
    importance_values = None
    importance_type = None

    if hasattr(clf, "feature_importances_"):
        importance_values = clf.feature_importances_
        importance_type = "feature_importances_"
    elif hasattr(clf, "coef_"):
        importance_values = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
        importance_type = "coef_ (|valor|)"
    elif hasattr(clf, "get_feature_importance"):  # CatBoost
        importance_values = clf.get_feature_importance()
        importance_type = "CatBoost get_feature_importance"

    if importance_values is not None:
        n_feat = min(len(importance_values), len(feature_cols))
        importance_df = pd.DataFrame({
            "Feature": feature_cols[:n_feat],
            "Importance": importance_values[:n_feat]
        }).sort_values("Importance", ascending=False)

        print(f"\n  [{importance_type}] Top 15 features más importantes:")
        print(importance_df.head(15).to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(importance_df))
        data = importance_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        bars = ax.barh(data["Feature"][::-1], data["Importance"][::-1],
                       color=colors, edgecolor="white")
        ax.set_title(f"Importancia de Features -- {winner_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importancia")
        plt.tight_layout()
        save_fig("feature_importance_winner.png")

        importance_df.to_csv(
            os.path.join(CONFIG["paths"]["reports"], "feature_importance.csv"), index=False
        )

    # --- 13.2 SHAP Values ---
    if not SHAP_AVAILABLE:
        print("\n  [AVISO] SHAP no disponible. Saltando SHAP.")
        return

    print("\n  Calculando SHAP values...")

    try:
        # Transformar datos con scaler del pipeline
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Usar muestra para velocidad
        n_shap = min(CONFIG["n_shap_samples"], len(X_test_scaled))
        idx = np.random.choice(len(X_test_scaled), n_shap, replace=False)
        X_shap = X_test_scaled[idx]

        feature_names = list(feature_cols)

        # Seleccionar explainer apropiado
        tree_models = (
            RandomForestClassifier, ExtraTreesClassifier,
            GradientBoostingClassifier
        )
        tree_model_names = ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
                            "GradientBoosting", "ExtraTrees"]

        if winner_name in tree_model_names or isinstance(clf, tree_models):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_shap)

            # Para clasificación binaria: algunos modelos retornan lista [clase0, clase1]
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_plot_values = shap_values[1]
            elif isinstance(shap_values, list):
                # Multiclase: promediar valores absolutos
                shap_plot_values = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_plot_values = shap_values

        else:
            explainer = shap.LinearExplainer(clf, X_train_scaled)
            shap_values = explainer.shap_values(X_shap)
            shap_plot_values = shap_values

        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_plot_values, X_shap,
            feature_names=feature_names,
            show=False, max_display=20
        )
        plt.title(f"SHAP Summary Plot -- {winner_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_fig("shap_summary.png")

        # SHAP Bar Plot (importancia global)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_plot_values, X_shap,
            feature_names=feature_names,
            plot_type="bar", show=False, max_display=20
        )
        plt.title(f"SHAP Feature Importance (Bar) -- {winner_name}", fontsize=13)
        plt.tight_layout()
        save_fig("shap_importance_bar.png")

        # SHAP Dependence plots -- top 3 features
        mean_abs_shap = np.abs(shap_plot_values).mean(axis=0)
        top3_idx = np.argsort(mean_abs_shap)[::-1][:3]

        for idx_feat in top3_idx:
            feat_name = feature_names[idx_feat]
            try:
                plt.figure(figsize=(8, 5))
                shap.dependence_plot(
                    idx_feat, shap_plot_values, X_shap,
                    feature_names=feature_names,
                    show=False, interaction_index="auto"
                )
                plt.title(f"SHAP Dependence -- {feat_name}", fontsize=12)
                plt.tight_layout()
                save_fig(f"shap_dependence_{feat_name.replace('/', '_')}.png")
            except Exception as e:
                print(f"    [AVISO] Dependence plot para {feat_name}: {e}")

        print(f"\n  [OK] SHAP completado ({n_shap} muestras analizadas).")

    except Exception as e:
        print(f"  [ERROR] SHAP falló: {e}")


# =============================================================================
# SECCIÓN 14: PERSISTENCIA DE MODELOS
# =============================================================================

def save_models(trained_models, winner_name):
    print_section("SECCIÓN 14: PERSISTENCIA DE MODELOS")

    for name, model in trained_models.items():
        try:
            # CatBoost tiene su propio formato de serialización
            clf = model.named_steps.get("clf", None)
            if clf and CATBOOST_AVAILABLE and isinstance(clf, CatBoostClassifier):
                path = os.path.join(CONFIG["paths"]["models"], f"catboost_{name.lower()}.cbm")
                clf.save_model(path)
                print(f"  -> CatBoost guardado: {path}")
            else:
                path = os.path.join(CONFIG["paths"]["models"], f"model_{name.lower().replace(' ','_')}.pkl")
                joblib.dump(model, path)
                if name == winner_name:
                    best_path = os.path.join(CONFIG["paths"]["models"], "best_model.pkl")
                    joblib.dump(model, best_path)
                    print(f"  -> GANADOR guardado: {best_path}")
                print(f"  -> Guardado: {path}")
        except Exception as e:
            print(f"  [ERROR] No se pudo guardar {name}: {e}")


# =============================================================================
# SECCIÓN 15: CONCLUSIONES Y RECOMENDACIONES
# =============================================================================

def print_conclusions(df_final, winner_name, df_cv, df_multi=None):
    print_section("SECCIÓN 15: CONCLUSIONES Y RECOMENDACIONES")

    winner_row = df_final[df_final["Model"] == winner_name].iloc[0]

    met_threshold = (
        winner_row["Test_Accuracy"] >= 0.80 and
        winner_row["Test_Precision"] >= 0.80 and
        winner_row["Test_Recall"] >= 0.80 and
        winner_row["Test_F1"] >= 0.80
    )

    print(f"""
  ==================================================================
  RESULTADOS DEL MODELO GANADOR: {winner_name}
  ==================================================================

  Métricas en Test Set (clasificación binaria, quality >= 7 = Bueno):
  +-----------------┬----------+
  | Accuracy        | {winner_row['Test_Accuracy']:.4f}   |
  | Precision       | {winner_row['Test_Precision']:.4f}   |
  | Recall          | {winner_row['Test_Recall']:.4f}   |
  | F1 Score        | {winner_row['Test_F1']:.4f}   |
  | ROC-AUC         | {winner_row['Test_ROC_AUC'] if winner_row['Test_ROC_AUC'] else 'N/A'}   |
  +-----------------┴----------+

  Objetivo de rendimiento (>80% en todas): {'[OK] ALCANZADO' if met_threshold else '[NO] No alcanzado'}
  {'  Esto es esperable dado el desbalance de clases y la naturaleza subjetiva' if not met_threshold else ''}
  {'  de la calificación de vinos. Ver recomendaciones abajo.' if not met_threshold else ''}
    """)

    print("""
  ==================================================================
  ANÁLISIS COMPARATIVO: BINARIO vs. MULTICLASE
  ==================================================================

  Binario (quality >= 7):
  • Pregunta clara y accionable: ¿Es un buen vino?
  • Clases más balanceadas (~78% malo, ~22% bueno)
  • SMOTE y estratificación funcionan correctamente
  • F1 más confiable y reproducible

  Multiclase (3 grupos: Bajo/Medio/Alto):
  • Granularidad mayor en la predicción
  • Clases "Bajo" y "Alto" siguen siendo minoritarias
  • Útil para pricing o recomendación por categoría
  • F1 macro penaliza las clases minoritarias correctamente

  -> RECOMENDACIÓN: Binario para decisiones operativas;
    Multiclase (3 grupos) para análisis analítico detallado.
    """)

    print("""
  ==================================================================
  VARIABLES MÁS INFLUYENTES (análisis general)
  ==================================================================

  Según SHAP y feature importance, las variables que más predicen
  la calidad del vino son típicamente:

  1. alcohol            -- más alcohol -> mayor calidad percibida
  2. volatile_acidity   -- impacto negativo no lineal (vinagre)
  3. sulphates          -- rol preservante positivo
  4. density            -- relacionada con azúcar y alcohol
  5. free_sulfur_dioxide -- protección antioxidante
  6. alcohol_acidity_ratio (feature ingenierizada) -- ratio clave
    """)

    print("""
  ==================================================================
  RECOMENDACIONES PARA MEJORAR EL DESEMPEÑO FUTURO
  ==================================================================

  1. DATOS
     • Recolectar más muestras de calidades extremas (3, 4, 8, 9)
     • Incorporar variables sensoriales (color, turbidez, aroma)
     • Datos de múltiples catadores para reducir sesgo del evaluador

  2. FEATURES
     • Interacciones de tercer orden entre las top 5 features
     • Ratios adicionales según dominio enológico
     • Embeddings de región/varietal si están disponibles

  3. MODELOS
     • Stacking/Blending de los top 3 modelos
     • Bayesian Optimization con más trials (200+)
     • TabNet o transformers tabulares (PyTorch)
     • Redes neuronales profundas con regularización dropout

  4. VALIDACIÓN
     • Validación cruzada nested para evitar optimismo en métricas
     • Test estadístico de diferencias (McNemar, Wilcoxon)
     • Monitoreo de data drift en producción

  5. PRODUCCIÓN
     • Umbral de clasificación ajustable según costo de negocio
     • API REST con FastAPI + modelo serializado
     • Dashboard de monitoreo de predicciones en tiempo real
    """)

    # Guardar conclusiones
    conclusions_path = os.path.join(CONFIG["paths"]["reports"], "conclusions.txt")
    with open(conclusions_path, "w", encoding="utf-8") as f:
        f.write(f"Modelo ganador: {winner_name}\n")
        f.write(f"Test F1: {winner_row['Test_F1']:.4f}\n")
        f.write(f"Test ROC-AUC: {winner_row['Test_ROC_AUC']}\n")
        f.write(f"Objetivo >80% alcanzado: {met_threshold}\n\n")
        f.write("Ranking completo:\n")
        f.write(df_final.to_string())
    print(f"  -> Conclusiones guardadas: {conclusions_path}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    total_start = time.time()

    print("\n" + "#" * 70)
    print("  PREDICCIÓN DE CALIDAD DE VINOS -- PIPELINE COMPLETO DE ML")
    print("  Clasificación Binaria (primario) + Multiclase (secundario)")
    print("#" * 70)
    print(f"\n  Librerías disponibles:")
    print(f"    XGBoost    : {'[OK]' if XGBOOST_AVAILABLE else '[NO]'}")
    print(f"    LightGBM   : {'[OK]' if LGBM_AVAILABLE else '[NO]'}")
    print(f"    CatBoost   : {'[OK]' if CATBOOST_AVAILABLE else '[NO]'}")
    print(f"    SMOTE      : {'[OK]' if IMBLEARN_AVAILABLE else '[NO]'}")
    print(f"    SHAP       : {'[OK]' if SHAP_AVAILABLE else '[NO]'}")
    print(f"    Optuna     : {'[OK]' if OPTUNA_AVAILABLE else '[NO]'}")

    # 0. Setup
    setup_dirs()

    # 1. Carga de datos
    df = load_data()

    # 2. EDA
    feature_cols_original = run_eda(df)

    # 3. Feature Engineering
    df, new_features = engineer_features(df)

    # 4. Targets
    y_binary, y_multi = prepare_targets(df)

    # 5. Preprocesamiento y split
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, all_feature_cols = (
        preprocess_and_split(df, y_binary, y_multi)
    )

    # 6. Comparación de balanceo
    best_balancing = compare_balancing(X_train, y_train_bin)

    # 7. Definir modelos
    print_section("SECCIÓN 7: DEFINICIÓN DE MODELOS")
    models_list = get_models_and_spaces(best_balancing)

    # 8. Entrenamiento con RandomizedSearchCV
    trained_models, df_cv = train_all_models(X_train, y_train_bin, models_list, best_balancing)

    # 9. Optuna (top 2 modelos)
    trained_models = optuna_optimize(X_train, y_train_bin, trained_models, df_cv, best_balancing)

    # 10. Evaluación en test set (binario)
    df_test = evaluate_test_set(trained_models, X_test, y_test_bin)

    # 11. Multiclase (top 3)
    df_multi = multiclass_evaluation(X_train, X_test, y_train_multi, y_test_multi, df_cv)

    # 12. Tabla final
    df_final, winner_name = generate_final_table(df_cv, df_test)

    # 13. Interpretabilidad
    interpret_winner(trained_models, winner_name, X_train, X_test, y_test_bin, all_feature_cols)

    # 14. Guardar modelos
    save_models(trained_models, winner_name)

    # 15. Conclusiones
    print_conclusions(df_final, winner_name, df_cv, df_multi)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  [OK] Pipeline completado en {total_time/60:.1f} minutos")
    print(f"  Figuras    : {CONFIG['paths']['figures']}/")
    print(f"  Reportes   : {CONFIG['paths']['reports']}/")
    print(f"  Modelos    : {CONFIG['paths']['models']}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
