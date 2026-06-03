"""
Predictor de Calidad de Vinos
Interfaz grafica con CustomTkinter
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import customtkinter as ctk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Configuracion de apariencia ──────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "best_model.pkl")

# ── Colores ──────────────────────────────────────────────────────────────────
COLOR_BG       = "#1a1a2e"
COLOR_PANEL    = "#16213e"
COLOR_ACCENT   = "#e94560"
COLOR_ACCENT2  = "#0f3460"
COLOR_TEXT     = "#eaeaea"
COLOR_SUBTEXT  = "#a0a0b0"
COLOR_GREEN    = "#2ecc71"
COLOR_RED      = "#e74c3c"
COLOR_CARD     = "#1e2a4a"

# ── Metricas del modelo (resultados del pipeline) ────────────────────────────
MODEL_METRICS = {
    "Modelo":           "XGBoost",
    "Accuracy":         0.8754,
    "Precision":        0.6767,
    "Recall":           0.7031,
    "F1 Score":         0.6897,
    "ROC-AUC":          0.9111,
    "Muestras Train":   5197,
    "Muestras Test":    1300,
    "Total Features":   22,
    "Balanceo":         "SMOTE",
    "CV Folds":         5,
    "CV F1 (media)":    0.6623,
    "CV F1 (std)":      0.0312,
}

FEATURE_IMPORTANCE = {
    "density x alcohol":       0.1677,
    "alcohol":                 0.0803,
    "volatile_acidity":        0.0702,
    "volatile_acidity^2":      0.0513,
    "citric_acid":             0.0455,
    "log(residual_sugar)":     0.0429,
    "density":                 0.0414,
    "residual_sugar":          0.0393,
    "log(free_SO2)":           0.0371,
    "sulphates":               0.0364,
    "chlorides":               0.0360,
    "free_SO2 / total_SO2":    0.0354,
    "pH x fixed_acidity":      0.0349,
    "sulphates / chlorides":   0.0349,
    "fixed_acidity":           0.0335,
}

# ── Rangos tipicos para los campos de entrada ────────────────────────────────
FIELDS = [
    ("fixed_acidity",        "Acidez fija",          "g/dm3",  "3.8",  "3.8 – 15.9",  float),
    ("volatile_acidity",     "Acidez volatil",        "g/dm3",  "0.28", "0.08 – 1.58", float),
    ("citric_acid",          "Acido citrico",          "g/dm3",  "0.30", "0.0 – 1.66",  float),
    ("residual_sugar",       "Azucar residual",       "g/dm3",  "3.0",  "0.6 – 65.8",  float),
    ("chlorides",            "Cloruros",              "g/dm3",  "0.047","0.009 – 0.611",float),
    ("free_sulfur_dioxide",  "SO2 libre",             "mg/dm3", "29.0", "1 – 289",      float),
    ("total_sulfur_dioxide", "SO2 total",             "mg/dm3", "118.0","6 – 440",      float),
    ("density",              "Densidad",              "g/cm3",  "0.9949","0.987 – 1.039",float),
    ("pH",                   "pH",                    "",       "3.21", "2.72 – 4.01",  float),
    ("sulphates",            "Sulfatos",              "g/dm3",  "0.51", "0.22 – 2.0",   float),
    ("alcohol",              "Alcohol",               "% vol",  "10.3", "8.0 – 14.9",   float),
]


# =============================================================================
# LOGICA DE PREDICCION
# =============================================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror(
            "Error",
            f"No se encontro el modelo en:\n{MODEL_PATH}\n\n"
            "Ejecuta main.py primero para generar el modelo."
        )
        return None
    return joblib.load(MODEL_PATH)


def compute_features(values: dict, wine_type: int) -> pd.DataFrame:
    """Calcula las 10 features ingenierizadas y arma el DataFrame completo."""
    eps = 1e-8
    fa  = values["fixed_acidity"]
    va  = values["volatile_acidity"]
    ca  = values["citric_acid"]
    rs  = values["residual_sugar"]
    cl  = values["chlorides"]
    fso = values["free_sulfur_dioxide"]
    tso = values["total_sulfur_dioxide"]
    den = values["density"]
    ph  = values["pH"]
    sul = values["sulphates"]
    alc = values["alcohol"]

    row = {
        "fixed_acidity":               fa,
        "volatile_acidity":            va,
        "citric_acid":                 ca,
        "residual_sugar":              rs,
        "chlorides":                   cl,
        "free_sulfur_dioxide":         fso,
        "total_sulfur_dioxide":        tso,
        "density":                     den,
        "pH":                          ph,
        "sulphates":                   sul,
        "alcohol":                     alc,
        "wine_type":                   wine_type,
        "alcohol_acidity_ratio":       alc / (fa + va + eps),
        "sulphates_alcohol_ratio":     sul / (alc + eps),
        "total_acidity":               fa + va + ca,
        "free_sulfur_ratio":           fso / (tso + eps),
        "log_residual_sugar":          np.log1p(rs),
        "log_free_so2":                np.log1p(fso),
        "density_alcohol_interaction": den * alc,
        "volatile_acidity_sq":         va ** 2,
        "sulphates_chlorides_ratio":   sul / (cl + eps),
        "pH_fixed_acidity_interaction":ph * fa,
    }
    return pd.DataFrame([row])


# =============================================================================
# PANTALLA PRINCIPAL (MENU)
# =============================================================================

class MainMenu(ctk.CTkFrame):
    def __init__(self, master, show_frame_cb):
        super().__init__(master, fg_color=COLOR_BG, corner_radius=0)
        self.show_frame = show_frame_cb
        self._build()

    def _build(self):
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Logo / titulo
        ctk.CTkLabel(
            self, text="?",
            font=("Segoe UI", 64),
            text_color=COLOR_ACCENT
        ).grid(row=0, pady=(40, 0))

        ctk.CTkLabel(
            self,
            text="Predictor de Calidad de Vinos",
            font=("Segoe UI", 28, "bold"),
            text_color=COLOR_TEXT
        ).grid(row=1, pady=(0, 4))

        ctk.CTkLabel(
            self,
            text="Modelo XGBoost  •  Accuracy 87.5%  •  ROC-AUC 91.1%",
            font=("Segoe UI", 13),
            text_color=COLOR_SUBTEXT
        ).grid(row=2, pady=(0, 40))

        # Botones de modulos
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, pady=10)

        self._menu_button(
            btn_frame,
            icon="?",
            title="Hacer Prediccion",
            subtitle="Ingresa variables fisicoquimicas\ny obtiene si el vino es Bueno o Malo",
            command=lambda: self.show_frame("prediccion"),
            col=0
        )

        self._menu_button(
            btn_frame,
            icon="?",
            title="Informacion del Modelo",
            subtitle="Metricas, features importantes\ny detalles tecnicos del pipeline",
            command=lambda: self.show_frame("info"),
            col=1
        )

        ctk.CTkLabel(
            self,
            text="Dataset: UCI Wine Quality  •  6,497 muestras  •  22 features",
            font=("Segoe UI", 11),
            text_color=COLOR_SUBTEXT
        ).grid(row=4, pady=(20, 10))

    def _menu_button(self, parent, icon, title, subtitle, command, col):
        card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=16, width=280, height=200)
        card.grid(row=0, column=col, padx=24, pady=10)
        card.grid_propagate(False)
        card.grid_rowconfigure((0, 1, 2, 3), weight=1)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text=icon, font=("Segoe UI", 40)).grid(row=0, pady=(20, 0))
        ctk.CTkLabel(card, text=title, font=("Segoe UI", 15, "bold"),
                     text_color=COLOR_TEXT).grid(row=1)
        ctk.CTkLabel(card, text=subtitle, font=("Segoe UI", 11),
                     text_color=COLOR_SUBTEXT, justify="center").grid(row=2)
        ctk.CTkButton(
            card, text="Abrir", width=140, height=34,
            fg_color=COLOR_ACCENT, hover_color="#c0392b",
            font=("Segoe UI", 13, "bold"),
            command=command
        ).grid(row=3, pady=(0, 18))


# =============================================================================
# MODULO DE PREDICCION
# =============================================================================

class PredictionModule(ctk.CTkFrame):
    def __init__(self, master, show_frame_cb, model):
        super().__init__(master, fg_color=COLOR_BG, corner_radius=0)
        self.show_frame = show_frame_cb
        self.model = model
        self.entries = {}
        self.wine_type_var = ctk.IntVar(value=1)
        self._build()

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ── Header ──
        header = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=0, height=60)
        header.grid(row=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        ctk.CTkButton(
            header, text="< Menu", width=90, height=34,
            fg_color="transparent", hover_color=COLOR_ACCENT2,
            font=("Segoe UI", 12), text_color=COLOR_SUBTEXT,
            command=lambda: self.show_frame("menu")
        ).grid(row=0, column=0, padx=16, pady=12)

        ctk.CTkLabel(
            header, text="? Hacer Prediccion",
            font=("Segoe UI", 18, "bold"), text_color=COLOR_TEXT
        ).grid(row=0, column=1, padx=10, sticky="w")

        # ── Contenido con scroll ──
        scroll = ctk.CTkScrollableFrame(self, fg_color=COLOR_BG, corner_radius=0)
        scroll.grid(row=2, sticky="nsew", padx=0, pady=0)
        scroll.grid_columnconfigure((0, 1), weight=1)

        # Tipo de vino
        wine_card = ctk.CTkFrame(scroll, fg_color=COLOR_CARD, corner_radius=12)
        wine_card.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(20, 10))
        wine_card.grid_columnconfigure((1, 2), weight=1)

        ctk.CTkLabel(wine_card, text="Tipo de vino",
                     font=("Segoe UI", 13, "bold"), text_color=COLOR_TEXT
                     ).grid(row=0, column=0, padx=20, pady=14)

        ctk.CTkRadioButton(
            wine_card, text="Blanco", variable=self.wine_type_var, value=1,
            font=("Segoe UI", 13), text_color=COLOR_TEXT,
            fg_color=COLOR_ACCENT
        ).grid(row=0, column=1, padx=10)

        ctk.CTkRadioButton(
            wine_card, text="Tinto", variable=self.wine_type_var, value=0,
            font=("Segoe UI", 13), text_color=COLOR_TEXT,
            fg_color=COLOR_ACCENT
        ).grid(row=0, column=2, padx=10)

        # Campos de variables
        for i, (key, label, unit, default, rng, _) in enumerate(FIELDS):
            row_i = (i // 2) + 1
            col_i = i % 2
            self._field_card(scroll, row_i, col_i, key, label, unit, default, rng)

        # Boton predecir
        btn_row = (len(FIELDS) // 2) + 2
        ctk.CTkButton(
            scroll, text="PREDECIR",
            font=("Segoe UI", 16, "bold"),
            height=52, corner_radius=12,
            fg_color=COLOR_ACCENT, hover_color="#c0392b",
            command=self._predict
        ).grid(row=btn_row, column=0, columnspan=2, padx=24, pady=16, sticky="ew")

        # Panel de resultado
        self.result_frame = ctk.CTkFrame(scroll, fg_color=COLOR_CARD, corner_radius=16)
        self.result_frame.grid(row=btn_row + 1, column=0, columnspan=2,
                                padx=24, pady=(0, 30), sticky="ew")
        self.result_frame.grid_columnconfigure(0, weight=1)

        self.result_icon  = ctk.CTkLabel(self.result_frame, text="",
                                          font=("Segoe UI", 52))
        self.result_label = ctk.CTkLabel(self.result_frame, text="",
                                          font=("Segoe UI", 22, "bold"))
        self.result_desc  = ctk.CTkLabel(self.result_frame, text="",
                                          font=("Segoe UI", 13),
                                          text_color=COLOR_SUBTEXT,
                                          wraplength=500, justify="center")
        self.result_prob  = ctk.CTkLabel(self.result_frame, text="",
                                          font=("Segoe UI", 13),
                                          text_color=COLOR_SUBTEXT)

        self.result_icon.grid(row=0, pady=(20, 0))
        self.result_label.grid(row=1)
        self.result_desc.grid(row=2, padx=30, pady=(4, 4))
        self.result_prob.grid(row=3, pady=(0, 20))

    def _field_card(self, parent, row, col, key, label, unit, default, rng):
        card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=12)
        card.grid(row=row, column=col, padx=(24 if col == 0 else 10, 10 if col == 0 else 24),
                  pady=6, sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text=label,
                     font=("Segoe UI", 12, "bold"), text_color=COLOR_TEXT,
                     anchor="w").grid(row=0, sticky="w", padx=14, pady=(10, 0))

        ctk.CTkLabel(card, text=f"Rango tipico: {rng}  {unit}",
                     font=("Segoe UI", 10), text_color=COLOR_SUBTEXT,
                     anchor="w").grid(row=1, sticky="w", padx=14)

        entry = ctk.CTkEntry(
            card, placeholder_text=default,
            font=("Segoe UI", 13), height=36,
            fg_color=COLOR_ACCENT2, border_color=COLOR_ACCENT2,
            text_color=COLOR_TEXT
        )
        entry.insert(0, default)
        entry.grid(row=2, sticky="ew", padx=14, pady=(4, 12))
        self.entries[key] = entry

    def _predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Modelo no cargado.")
            return

        values = {}
        for key, label, unit, default, rng, dtype in FIELDS:
            raw = self.entries[key].get().strip().replace(",", ".")
            try:
                values[key] = dtype(raw)
            except ValueError:
                messagebox.showerror(
                    "Valor invalido",
                    f"El campo '{label}' tiene un valor no valido: '{raw}'"
                )
                return

        wine_type = self.wine_type_var.get()
        X = compute_features(values, wine_type)

        try:
            pred   = self.model.predict(X)[0]
            proba  = self.model.predict_proba(X)[0]
            prob_good = proba[1] * 100
            prob_bad  = proba[0] * 100
        except Exception as e:
            messagebox.showerror("Error en prediccion", str(e))
            return

        if pred == 1:
            self.result_icon.configure(text="?", text_color=COLOR_GREEN)
            self.result_label.configure(text="VINO BUENO", text_color=COLOR_GREEN)
            self.result_desc.configure(
                text="El modelo predice que este vino tiene calidad >= 7.\n"
                     "Es un vino de buena calidad segun sus caracteristicas fisicoquimicas."
            )
        else:
            self.result_icon.configure(text="?", text_color=COLOR_RED)
            self.result_label.configure(text="VINO MALO", text_color=COLOR_RED)
            self.result_desc.configure(
                text="El modelo predice que este vino tiene calidad < 7.\n"
                     "Las caracteristicas fisicoquimicas no alcanzan el umbral de buena calidad."
            )

        wine_name = "Blanco" if wine_type == 1 else "Tinto"
        self.result_prob.configure(
            text=f"Vino {wine_name}  |  Probabilidad Bueno: {prob_good:.1f}%  |  Probabilidad Malo: {prob_bad:.1f}%"
        )


# =============================================================================
# MODULO DE INFORMACION DEL MODELO
# =============================================================================

class ModelInfoModule(ctk.CTkFrame):
    def __init__(self, master, show_frame_cb):
        super().__init__(master, fg_color=COLOR_BG, corner_radius=0)
        self.show_frame = show_frame_cb
        self._build()

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ── Header ──
        header = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=0, height=60)
        header.grid(row=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        ctk.CTkButton(
            header, text="< Menu", width=90, height=34,
            fg_color="transparent", hover_color=COLOR_ACCENT2,
            font=("Segoe UI", 12), text_color=COLOR_SUBTEXT,
            command=lambda: self.show_frame("menu")
        ).grid(row=0, column=0, padx=16, pady=12)

        ctk.CTkLabel(
            header, text="? Informacion del Modelo",
            font=("Segoe UI", 18, "bold"), text_color=COLOR_TEXT
        ).grid(row=0, column=1, padx=10, sticky="w")

        # ── Tabs ──
        tab = ctk.CTkTabview(self, fg_color=COLOR_PANEL,
                              segmented_button_fg_color=COLOR_ACCENT2,
                              segmented_button_selected_color=COLOR_ACCENT,
                              segmented_button_selected_hover_color="#c0392b",
                              text_color=COLOR_TEXT)
        tab.grid(row=2, sticky="nsew", padx=20, pady=16)
        tab.grid_columnconfigure(0, weight=1)

        tab.add("Metricas")
        tab.add("Features")
        tab.add("Pipeline")
        tab.add("Acerca de")

        self._tab_metricas(tab.tab("Metricas"))
        self._tab_features(tab.tab("Features"))
        self._tab_pipeline(tab.tab("Pipeline"))
        self._tab_about(tab.tab("Acerca de"))

    # ── Tab: Metricas ────────────────────────────────────────────────────────
    def _tab_metricas(self, parent):
        parent.grid_columnconfigure((0, 1), weight=1)

        # Cards de metricas principales
        metrics_cards = [
            ("Accuracy",   MODEL_METRICS["Accuracy"],   "#3498db", "Porcentaje total de aciertos"),
            ("Precision",  MODEL_METRICS["Precision"],  "#e67e22", "Cuando dice 'Bueno', ¿cuantas veces acierta?"),
            ("Recall",     MODEL_METRICS["Recall"],     "#9b59b6", "¿Cuantos buenos reales encontro?"),
            ("F1 Score",   MODEL_METRICS["F1 Score"],   "#2ecc71", "Equilibrio entre Precision y Recall"),
            ("ROC-AUC",    MODEL_METRICS["ROC-AUC"],    "#e74c3c", "Capacidad de separar clases (0.91 = excelente)"),
            ("CV F1",      MODEL_METRICS["CV F1 (media)"], "#1abc9c", "F1 promedio en validacion cruzada (5 folds)"),
        ]

        for i, (name, val, color, desc) in enumerate(metrics_cards):
            row, col = divmod(i, 2)
            card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=14)
            card.grid(row=row, column=col, padx=10, pady=8, sticky="ew")
            card.grid_columnconfigure(0, weight=1)

            # Barra de progreso
            ctk.CTkLabel(card, text=name,
                         font=("Segoe UI", 12, "bold"), text_color=COLOR_TEXT
                         ).grid(row=0, sticky="w", padx=16, pady=(14, 0))
            ctk.CTkLabel(card, text=f"{val*100:.1f}%",
                         font=("Segoe UI", 26, "bold"), text_color=color
                         ).grid(row=1, sticky="w", padx=16)
            ctk.CTkProgressBar(card, progress_color=color,
                                fg_color=COLOR_ACCENT2, height=8, corner_radius=4
                                ).grid(row=2, sticky="ew", padx=16, pady=(4, 0))
            ctk.CTkLabel(card, text=desc,
                         font=("Segoe UI", 10), text_color=COLOR_SUBTEXT
                         ).grid(row=3, sticky="w", padx=16, pady=(2, 12))

            # Actualizar barra
            for w in card.winfo_children():
                if isinstance(w, ctk.CTkProgressBar):
                    w.set(val)

        # Datos adicionales
        extra = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=14)
        extra.grid(row=3, column=0, columnspan=2, padx=10, pady=8, sticky="ew")
        extra.grid_columnconfigure((0, 1, 2, 3), weight=1)

        items = [
            ("Train", str(MODEL_METRICS["Muestras Train"])),
            ("Test",  str(MODEL_METRICS["Muestras Test"])),
            ("Features", str(MODEL_METRICS["Total Features"])),
            ("Balanceo", MODEL_METRICS["Balanceo"]),
        ]
        for i, (label, val) in enumerate(items):
            ctk.CTkLabel(extra, text=val,
                         font=("Segoe UI", 20, "bold"), text_color=COLOR_ACCENT
                         ).grid(row=0, column=i, padx=20, pady=(16, 0))
            ctk.CTkLabel(extra, text=label,
                         font=("Segoe UI", 11), text_color=COLOR_SUBTEXT
                         ).grid(row=1, column=i, padx=20, pady=(0, 14))

    # ── Tab: Features ────────────────────────────────────────────────────────
    def _tab_features(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_PANEL)
        ax.set_facecolor(COLOR_PANEL)

        names = list(FEATURE_IMPORTANCE.keys())[::-1]
        vals  = list(FEATURE_IMPORTANCE.values())[::-1]

        colors = [COLOR_ACCENT if v == max(FEATURE_IMPORTANCE.values()) else "#4a90d9"
                  for v in vals]

        bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.65)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{v*100:.1f}%", va="center", color=COLOR_TEXT, fontsize=8.5)

        ax.set_xlabel("Importancia", color=COLOR_SUBTEXT, fontsize=10)
        ax.set_title("Importancia de Features — XGBoost", color=COLOR_TEXT,
                     fontsize=13, fontweight="bold", pad=14)
        ax.tick_params(colors=COLOR_TEXT, labelsize=9)
        ax.spines[:].set_color("#2a3a5e")
        ax.xaxis.label.set_color(COLOR_SUBTEXT)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, sticky="nsew", padx=10, pady=10)
        plt.close(fig)

    # ── Tab: Pipeline ────────────────────────────────────────────────────────
    def _tab_pipeline(self, parent):
        parent.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.grid(row=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        steps = [
            ("1", "Carga de Datos",
             "Se combinan vino_rojo.csv y vino_blanco.csv\n"
             "• 1,599 vinos tintos  +  4,898 vinos blancos = 6,497 total\n"
             "• Se agrega columna 'wine_type' (0=Tinto, 1=Blanco)\n"
             "• Separador: punto y coma (;)"),

            ("2", "Analisis Exploratorio (EDA)",
             "• Sin valores faltantes\n"
             "• Variable objetivo: quality (3-9), muy concentrada en 5-6\n"
             "• Top correlacion con quality: alcohol (0.44), density (-0.31)\n"
             "• Outliers detectados por IQR en citric_acid (7.8%) y volatile_acidity (5.8%)"),

            ("3", "Ingenieria de Caracteristicas",
             "Se crearon 10 variables nuevas:\n"
             "• density x alcohol        — sensacion en boca y cuerpo\n"
             "• volatile_acidity^2       — impacto no lineal del sabor a vinagre\n"
             "• alcohol / total_acidity  — balance de madurez de la uva\n"
             "• log(residual_sugar)      — normaliza distribucion sesgada\n"
             "• free_SO2 / total_SO2     — proporcion de SO2 activo\n"
             "• sulphates / chlorides    — balance preservacion vs salinidad\n"
             "• + 4 features adicionales"),

            ("4", "Preprocesamiento",
             "• Winsorizacion al percentil 1 y 99 (evita outliers extremos)\n"
             "• Split estratificado: 80% train (5,197) / 20% test (1,300)\n"
             "• StandardScaler dentro del pipeline (evita data leakage)\n"
             "• Target binario: quality >= 7 = Bueno (19.7%), < 7 = Malo (80.3%)"),

            ("5", "Balanceo de Clases",
             "Comparacion de 3 estrategias con Random Forest (CV 5-fold):\n"
             "• Sin balanceo:         F1 = 0.638\n"
             "• class_weight=balanced: F1 = 0.627\n"
             "• SMOTE (ganador):       F1 = 0.669\n\n"
             "SMOTE genera muestras sinteticas SOLO dentro de cada fold\n"
             "usando imblearn.pipeline.Pipeline para evitar data leakage."),

            ("6", "Entrenamiento y Seleccion",
             "9 modelos evaluados con RandomizedSearchCV (50 iter, 5-fold):\n"
             "• ExtraTrees       CV F1 = 0.684  *\n"
             "• RandomForest     CV F1 = 0.671\n"
             "• LightGBM         CV F1 = 0.670\n"
             "• XGBoost          CV F1 = 0.662\n"
             "• GradientBoosting CV F1 = 0.655\n"
             "• MLPClassifier    CV F1 = 0.641\n"
             "• SVC              CV F1 = 0.614\n"
             "• LogisticReg.     CV F1 = 0.532\n\n"
             "Optuna (80 trials) afino ExtraTrees y RandomForest."),

            ("7", "Modelo Final: XGBoost",
             "XGBoost gano en el test set:\n"
             "• Accuracy  : 87.54%\n"
             "• F1 Score  : 68.97%\n"
             "• ROC-AUC   : 91.11%  <- discriminacion excelente\n\n"
             "Parametros optimizados:\n"
             "• learning_rate: 0.0396\n"
             "• max_depth: 9\n"
             "• n_estimators: 416\n"
             "• subsample: 0.884\n"
             "• colsample_bytree: 0.675"),
        ]

        for i, (num, title, desc) in enumerate(steps):
            card = ctk.CTkFrame(scroll, fg_color=COLOR_CARD, corner_radius=12)
            card.grid(row=i, sticky="ew", padx=10, pady=6)
            card.grid_columnconfigure(1, weight=1)

            # Numero del paso
            badge = ctk.CTkFrame(card, fg_color=COLOR_ACCENT, corner_radius=8,
                                  width=32, height=32)
            badge.grid(row=0, column=0, padx=(14, 10), pady=14, sticky="n")
            badge.grid_propagate(False)
            ctk.CTkLabel(badge, text=num,
                         font=("Segoe UI", 12, "bold"),
                         text_color="white").place(relx=0.5, rely=0.5, anchor="center")

            ctk.CTkLabel(card, text=title,
                         font=("Segoe UI", 13, "bold"), text_color=COLOR_TEXT,
                         anchor="w").grid(row=0, column=1, sticky="w", pady=(14, 0))
            ctk.CTkLabel(card, text=desc,
                         font=("Segoe UI", 11), text_color=COLOR_SUBTEXT,
                         anchor="w", justify="left", wraplength=560
                         ).grid(row=1, column=1, sticky="w", padx=(0, 14), pady=(0, 14))

    # ── Tab: Acerca de ───────────────────────────────────────────────────────
    def _tab_about(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        ctk.CTkLabel(frame, text="?",
                     font=("Segoe UI", 60), text_color=COLOR_ACCENT
                     ).grid(row=0, pady=(30, 0))

        ctk.CTkLabel(frame, text="Predictor de Calidad de Vinos",
                     font=("Segoe UI", 22, "bold"), text_color=COLOR_TEXT
                     ).grid(row=1)

        ctk.CTkLabel(
            frame,
            text="Proyecto de Machine Learning\n"
                 "Dataset: UCI Wine Quality  (P. Cortez et al., 2009)\n"
                 "Modelo: XGBoost con SMOTE + StandardScaler\n"
                 "Umbral de clasificacion: quality >= 7 = Bueno",
            font=("Segoe UI", 13), text_color=COLOR_SUBTEXT,
            justify="center"
        ).grid(row=2, pady=10)

        info_frame = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=14)
        info_frame.grid(row=3, padx=60, pady=10, sticky="ew")
        info_frame.grid_columnconfigure((0, 1, 2), weight=1)

        for i, (k, v) in enumerate([
            ("Lenguaje",   "Python 3.11"),
            ("Framework",  "scikit-learn"),
            ("GUI",        "CustomTkinter"),
        ]):
            ctk.CTkLabel(info_frame, text=v,
                         font=("Segoe UI", 14, "bold"), text_color=COLOR_ACCENT
                         ).grid(row=0, column=i, padx=20, pady=(16, 0))
            ctk.CTkLabel(info_frame, text=k,
                         font=("Segoe UI", 11), text_color=COLOR_SUBTEXT
                         ).grid(row=1, column=i, padx=20, pady=(0, 14))

        ctk.CTkLabel(
            frame,
            text="Nota: El F1 de ~69% es esperable — la calidad del vino es subjetiva\n"
                 "y los estudios academicos con este dataset reportan entre 65-75%.",
            font=("Segoe UI", 11), text_color=COLOR_SUBTEXT, justify="center"
        ).grid(row=4, pady=(0, 20))


# =============================================================================
# APLICACION PRINCIPAL
# =============================================================================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Predictor de Calidad de Vinos")
        self.geometry("860x720")
        self.minsize(760, 620)
        self.configure(fg_color=COLOR_BG)

        # Cargar modelo
        self.model = load_model()

        # Construir frames
        self.frames = {}
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.frames["menu"]      = MainMenu(self, self.show_frame)
        self.frames["prediccion"] = PredictionModule(self, self.show_frame, self.model)
        self.frames["info"]      = ModelInfoModule(self, self.show_frame)

        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("menu")

    def show_frame(self, name: str):
        self.frames[name].tkraise()


# =============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
