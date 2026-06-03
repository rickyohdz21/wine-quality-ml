const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  TableOfContents
} = require("docx");
const fs = require("fs");

// ─── Colores ──────────────────────────────────────────────────────────────────
const C_WINE   = "7B2D42";
const C_DARK   = "2C3E50";
const C_BLUE   = "2980B9";
const C_GREEN  = "27AE60";
const C_ORANGE = "E67E22";
const C_GRAY_H = "EAF0FB";
const C_GRAY_R = "F7F9FC";
const C_WHITE  = "FFFFFF";
const C_BORDER = "B8C4D4";
const CONTENT_WIDTH = 9360;

// ─── Helpers de tabla ─────────────────────────────────────────────────────────
const border     = { style: BorderStyle.SINGLE, size: 1, color: C_BORDER };
const borders    = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };

function thCell(text, w, span) {
  const cfg = {
    borders, verticalAlign: VerticalAlign.CENTER,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: C_WINE, type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, size: 20, color: C_WHITE, font: "Arial" })]
    })]
  };
  if (span) cfg.columnSpan = span;
  return new TableCell(cfg);
}

function tdCell(text, w, shade = false, align = AlignmentType.LEFT, isBold = false) {
  return new TableCell({
    borders, verticalAlign: VerticalAlign.CENTER,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: shade ? C_GRAY_R : C_WHITE, type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, size: 20, font: "Arial", color: C_DARK, bold: isBold })]
    })]
  });
}

function tdGreen(text, w) {
  return new TableCell({
    borders, verticalAlign: VerticalAlign.CENTER,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: "D5F5E3", type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({
      alignment: AlignmentType.LEFT,
      children: [new TextRun({ text, size: 20, font: "Arial", color: C_GREEN, bold: true })]
    })]
  });
}

// ─── Helpers de párrafo ───────────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: C_WINE, space: 6 } },
    children: [new TextRun({ text, bold: true, size: 34, color: C_WINE, font: "Arial" })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 100 },
    children: [new TextRun({ text, bold: true, size: 28, color: C_DARK, font: "Arial" })]
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, bold: true, size: 24, color: C_BLUE, font: "Arial" })]
  });
}

function p(runs, after = 120) {
  const children = typeof runs === "string"
    ? [new TextRun({ text: runs, size: 22, font: "Arial", color: C_DARK })]
    : runs;
  return new Paragraph({ children, spacing: { after } });
}

function b(text, color = C_DARK) {
  return new TextRun({ text, bold: true, size: 22, font: "Arial", color });
}
function n(text, color = C_DARK) {
  return new TextRun({ text, size: 22, font: "Arial", color });
}
function it(text) {
  return new TextRun({ text, italics: true, size: 22, font: "Arial", color: C_DARK });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, font: "Arial", color: C_DARK })]
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, font: "Arial", color: C_DARK })]
  });
}

function spacer() {
  return new Paragraph({ children: [new TextRun("")], spacing: { after: 80 } });
}

function pb() {
  return new Paragraph({ children: [new PageBreak()] });
}

function callout(title, text, color = C_BLUE) {
  const rows = [];
  if (title) {
    rows.push(new TableRow({ children: [new TableCell({
      borders, width: { size: CONTENT_WIDTH, type: WidthType.DXA },
      shading: { fill: color, type: ShadingType.CLEAR }, margins: cellMargins,
      children: [new Paragraph({ children: [new TextRun({ text: title, bold: true, size: 22, color: C_WHITE, font: "Arial" })] })]
    })] }));
  }
  rows.push(new TableRow({ children: [new TableCell({
    borders, width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    shading: { fill: C_GRAY_H, type: ShadingType.CLEAR },
    margins: { top: 120, bottom: 120, left: 160, right: 160 },
    children: [new Paragraph({ children: [new TextRun({ text, size: 21, font: "Arial", color: C_DARK })] })]
  })] }));
  return new Table({ width: { size: CONTENT_WIDTH, type: WidthType.DXA }, columnWidths: [CONTENT_WIDTH], rows });
}

// ─── DOCUMENTO ────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }
        ]
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
        ]
      }
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22, color: C_DARK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, font: "Arial", color: C_WINE },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: C_DARK },
        paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: C_BLUE },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } }
    ]
  },

  sections: [

    // ══════════════════════════════════════════════════════════════════════════
    // PORTADA
    // ══════════════════════════════════════════════════════════════════════════
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      children: [
        spacer(), spacer(), spacer(), spacer(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
          children: [new TextRun({ text: "PROYECTO DE MACHINE LEARNING", size: 48, bold: true, font: "Arial", color: C_WINE })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: C_WINE, space: 8 } },
          children: [new TextRun({ text: "Predicción de Calidad de Vinos", size: 64, bold: true, font: "Arial", color: C_DARK })]
        }),
        spacer(), spacer(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "Documentación Técnica Completa", size: 28, italics: true, font: "Arial", color: C_BLUE })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "Pipeline de Machine Learning: EDA, Ingeniería de Features, Modelos, Optimización e Interpretabilidad", size: 22, italics: true, font: "Arial", color: "666666" })]
        }),
        spacer(), spacer(), spacer(), spacer(), spacer(), spacer(),
        new Table({
          width: { size: 6000, type: WidthType.DXA }, columnWidths: [3000, 3000],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: C_WINE, type: ShadingType.CLEAR }, margins: cellMargins,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Modelo Ganador", bold: true, size: 20, color: C_WHITE, font: "Arial" })] })] }),
              new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: C_GRAY_H, type: ShadingType.CLEAR }, margins: cellMargins,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "XGBoost", bold: true, size: 20, color: C_DARK, font: "Arial" })] })] })
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: C_WINE, type: ShadingType.CLEAR }, margins: cellMargins,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Accuracy", bold: true, size: 20, color: C_WHITE, font: "Arial" })] })] }),
              new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: C_GRAY_H, type: ShadingType.CLEAR }, margins: cellMargins,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "87.54 %", bold: true, size: 20, color: C_DARK, font: "Arial" })] })] })
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: C_WINE, type: ShadingType.CLEAR }, margins: cellMargins,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "ROC-AUC", bold: true, size: 20, color: C_WHITE, font: "Arial" })] })] }),
              new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: C_GRAY_H, type: ShadingType.CLEAR }, margins: cellMargins,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "91.11 %", bold: true, size: 20, color: C_DARK, font: "Arial" })] })] })
            ]})
          ]
        }),
        spacer(), spacer(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Mayo 2026", size: 22, font: "Arial", color: "888888" })]
        }),
        pb()
      ]
    },

    // ══════════════════════════════════════════════════════════════════════════
    // CUERPO PRINCIPAL
    // ══════════════════════════════════════════════════════════════════════════
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 } }
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C_WINE, space: 4 } },
          children: [
            new TextRun({ text: "Proyecto de Machine Learning: Predicción de Calidad de Vinos", size: 18, font: "Arial", color: "888888" }),
            new TextRun({ text: "   |   Documentación Técnica Completa", size: 18, font: "Arial", color: C_WINE })
          ]
        })] })
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 6, color: C_WINE, space: 4 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Página ", size: 18, font: "Arial", color: "888888" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: C_WINE }),
            new TextRun({ text: " de ", size: 18, font: "Arial", color: "888888" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, font: "Arial", color: "888888" })
          ]
        })] })
      },

      children: [

        // ── Tabla de contenidos ───────────────────────────────────────────
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 0, after: 200 },
          children: [new TextRun({ text: "TABLA DE CONTENIDOS", size: 32, bold: true, font: "Arial", color: C_WINE })]
        }),
        new TableOfContents("Tabla de Contenidos", { hyperlink: true, headingStyleRange: "1-3" }),
        pb(),

        // ══════════════════════════════════════════════════════════════════
        // 1. INTRODUCCIÓN
        // ══════════════════════════════════════════════════════════════════
        h1("1. Introducción y Objetivo del Proyecto"),
        p([n("Este documento describe el desarrollo completo de un proyecto de "), b("Machine Learning (ML)"),
           n(" aplicado a la predicción de la calidad de vinos a partir de sus propiedades fisicoquímicas. El proyecto combina buenas prácticas de ciencia de datos con técnicas modernas de optimización, interpretabilidad y despliegue.")]),

        h2("1.1 ¿Qué problema resolvemos?"),
        p("Un productor o enólogo recibe muestras de vino con mediciones de laboratorio (acidez, alcohol, sulfatos, etc.) y quiere saber rápidamente si ese vino tiene buena calidad o no, sin depender exclusivamente de un catador humano que puede ser subjetivo y costoso."),
        p("El modelo de Machine Learning aprende esos patrones a partir de miles de muestras históricas y puede hacer la clasificación en milisegundos."),

        h2("1.2 Enfoque adoptado"),
        p("Se analizaron dos enfoques posibles:"),
        bullet("Clasificación Binaria: el vino es Bueno (quality ≥ 7) o Malo (quality < 7)."),
        bullet("Clasificación Multiclase: predecir la puntuación exacta de 3 a 9."),
        spacer(),
        callout(
          "Decisión metodológica: ¿por qué elegimos el enfoque Binario como primario?",
          "La variable quality tiene 7 valores distintos (3 a 9), pero los extremos (3, 4, 8, 9) tienen muy pocas muestras. " +
          "Por ejemplo, la calidad 9 solo tiene 5 registros en todo el dataset. Entrenar con tan pocos ejemplos " +
          "genera modelos poco confiables para esas clases. Además, la técnica SMOTE de balanceo " +
          "necesita al menos 6 muestras por clase para funcionar. El enfoque binario resuelve esto " +
          "con un umbral claro (≥ 7 = Bueno) y es una pregunta de negocio directa y útil.",
          C_WINE
        ),
        spacer(),

        h2("1.3 Dataset utilizado"),
        p("Se utilizaron dos archivos del repositorio UCI Wine Quality Dataset:"),
        bullet("winequality-red.csv: 1 599 registros de vino tinto"),
        bullet("winequality-white.csv: 4 898 registros de vino blanco"),
        bullet("Total combinado: 6 497 muestras con 11 variables fisicoquímicas + 1 variable objetivo (quality)"),
        spacer(),

        // ══════════════════════════════════════════════════════════════════
        // 2. DESCRIPCIÓN DEL DATASET
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("2. Descripción del Dataset"),

        h2("2.1 Variables fisicoquímicas (features)"),
        p("Cada muestra de vino está descrita por las siguientes 11 variables de laboratorio:"),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [2600, 1700, 5060],
          rows: [
            new TableRow({ children: [thCell("Variable", 2600), thCell("Unidad", 1700), thCell("Descripción", 5060)] }),
            new TableRow({ children: [tdCell("fixed_acidity", 2600, false, AlignmentType.LEFT, true), tdCell("g/dm³", 1700, false, AlignmentType.CENTER), tdCell("Acidez fija: cantidad de ácido tartárico. Aporta estabilidad microbiológica al vino.", 5060)] }),
            new TableRow({ children: [tdCell("volatile_acidity", 2600, true, AlignmentType.LEFT, true), tdCell("g/dm³", 1700, true, AlignmentType.CENTER), tdCell("Acidez volátil: cantidad de ácido acético. Niveles altos generan sabor a vinagre, lo que deteriora la calidad.", 5060, true)] }),
            new TableRow({ children: [tdCell("citric_acid", 2600, false, AlignmentType.LEFT, true), tdCell("g/dm³", 1700, false, AlignmentType.CENTER), tdCell("Ácido cítrico: en pequeñas cantidades aporta frescura y sabor al vino.", 5060)] }),
            new TableRow({ children: [tdCell("residual_sugar", 2600, true, AlignmentType.LEFT, true), tdCell("g/dm³", 1700, true, AlignmentType.CENTER), tdCell("Azúcar residual: azúcar que queda tras la fermentación. Los vinos blancos tienen mucho más azúcar que los tintos.", 5060, true)] }),
            new TableRow({ children: [tdCell("chlorides", 2600, false, AlignmentType.LEFT, true), tdCell("g/dm³", 1700, false, AlignmentType.CENTER), tdCell("Cloruros: contenido de sal. Valores muy altos afectan negativamente el sabor.", 5060)] }),
            new TableRow({ children: [tdCell("free_sulfur_dioxide", 2600, true, AlignmentType.LEFT, true), tdCell("mg/dm³", 1700, true, AlignmentType.CENTER), tdCell("Dióxido de azufre libre: forma activa del SO₂. Actúa como antioxidante y antimicrobiano.", 5060, true)] }),
            new TableRow({ children: [tdCell("total_sulfur_dioxide", 2600, false, AlignmentType.LEFT, true), tdCell("mg/dm³", 1700, false, AlignmentType.CENTER), tdCell("Dióxido de azufre total: suma del SO₂ libre y el ligado (inactivo). Regulado por ley.", 5060)] }),
            new TableRow({ children: [tdCell("density", 2600, true, AlignmentType.LEFT, true), tdCell("g/cm³", 1700, true, AlignmentType.CENTER), tdCell("Densidad: relacionada con el contenido de alcohol y azúcar. Vinos más dulces tienen mayor densidad.", 5060, true)] }),
            new TableRow({ children: [tdCell("pH", 2600, false, AlignmentType.LEFT, true), tdCell("—", 1700, false, AlignmentType.CENTER), tdCell("pH: mide la acidez total del vino en escala logarítmica. La mayoría de los vinos tienen pH entre 3 y 4.", 5060)] }),
            new TableRow({ children: [tdCell("sulphates", 2600, true, AlignmentType.LEFT, true), tdCell("g/dm³", 1700, true, AlignmentType.CENTER), tdCell("Sulfatos: aditivo que contribuye a los niveles de SO₂. Tienen efecto antimicrobiano y antioxidante.", 5060, true)] }),
            new TableRow({ children: [tdCell("alcohol", 2600, false, AlignmentType.LEFT, true), tdCell("% vol", 1700, false, AlignmentType.CENTER), tdCell("Porcentaje de alcohol. Es la variable con mayor correlación positiva con la calidad (r = 0.44).", 5060)] })
          ]
        }),
        spacer(),
        p([it("Variable objetivo: "), n("quality (entero del 3 al 9) — puntuación asignada por catadores humanos.")]),

        h2("2.2 Distribución de la variable objetivo"),
        p("La distribución de quality en el dataset combinado es muy desequilibrada:"),
        spacer(),

        new Table({
          width: { size: 7200, type: WidthType.DXA },
          columnWidths: [1440, 1440, 1440, 1440, 1440],
          rows: [
            new TableRow({ children: [thCell("Quality", 1440), thCell("Muestras", 1440), thCell("Porcentaje", 1440), thCell("Tipo de vino", 1440), thCell("Clasificación binaria", 1440)] }),
            new TableRow({ children: [tdCell("3", 1440, false, AlignmentType.CENTER), tdCell("30", 1440, false, AlignmentType.CENTER), tdCell("0.5 %", 1440, false, AlignmentType.CENTER), tdCell("Ambos", 1440, false, AlignmentType.CENTER), tdCell("Malo", 1440, false, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("4", 1440, true, AlignmentType.CENTER), tdCell("216", 1440, true, AlignmentType.CENTER), tdCell("3.3 %", 1440, true, AlignmentType.CENTER), tdCell("Ambos", 1440, true, AlignmentType.CENTER), tdCell("Malo", 1440, true, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("5", 1440, false, AlignmentType.CENTER), tdCell("2 138", 1440, false, AlignmentType.CENTER), tdCell("32.9 %", 1440, false, AlignmentType.CENTER), tdCell("Ambos", 1440, false, AlignmentType.CENTER), tdCell("Malo", 1440, false, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("6", 1440, true, AlignmentType.CENTER), tdCell("2 836", 1440, true, AlignmentType.CENTER), tdCell("43.7 %", 1440, true, AlignmentType.CENTER), tdCell("Ambos", 1440, true, AlignmentType.CENTER), tdCell("Malo", 1440, true, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("7", 1440, false, AlignmentType.CENTER), tdCell("1 079", 1440, false, AlignmentType.CENTER), tdCell("16.6 %", 1440, false, AlignmentType.CENTER), tdCell("Ambos", 1440, false, AlignmentType.CENTER), tdCell("BUENO", 1440, false, AlignmentType.CENTER, true)] }),
            new TableRow({ children: [tdCell("8", 1440, true, AlignmentType.CENTER), tdCell("193", 1440, true, AlignmentType.CENTER), tdCell("3.0 %", 1440, true, AlignmentType.CENTER), tdCell("Ambos", 1440, true, AlignmentType.CENTER), tdCell("BUENO", 1440, true, AlignmentType.CENTER, true)] }),
            new TableRow({ children: [tdCell("9", 1440, false, AlignmentType.CENTER), tdCell("5", 1440, false, AlignmentType.CENTER), tdCell("0.1 %", 1440, false, AlignmentType.CENTER), tdCell("Solo blanco", 1440, false, AlignmentType.CENTER), tdCell("BUENO", 1440, false, AlignmentType.CENTER, true)] }),
            new TableRow({ children: [tdCell("TOTAL", 1440, true, AlignmentType.CENTER, AlignmentType.CENTER, true), tdCell("6 497", 1440, true, AlignmentType.CENTER, AlignmentType.CENTER, true), tdCell("100 %", 1440, true, AlignmentType.CENTER, AlignmentType.CENTER, true), tdCell("—", 1440, true, AlignmentType.CENTER), tdCell("—", 1440, true, AlignmentType.CENTER)] })
          ]
        }),
        spacer(),
        p("Conclusión: el 80.3 % de los vinos son clasificados como Malo y solo el 19.7 % como Bueno. Este desbalance es el principal reto que se debe abordar con técnicas especiales."),

        // ══════════════════════════════════════════════════════════════════
        // 3. EDA
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("3. Análisis Exploratorio de Datos (EDA)"),
        p("El Análisis Exploratorio de Datos (EDA) es el primer paso obligatorio en cualquier proyecto de Machine Learning. Su objetivo es entender los datos antes de modelar: detectar problemas, comprender distribuciones y encontrar relaciones entre variables."),

        h2("3.1 Calidad de los datos"),
        bullet("Valores faltantes: NINGUNO. Todos los 6 497 registros están completos."),
        bullet("Tipos de datos: todas las variables son numéricas (float64 o int64). No se requiere codificación categórica."),
        bullet("Duplicados: se mantuvo el dataset original sin eliminar duplicados, ya que pueden representar vinos con características fisicoquímicas idénticas pero distintas calificaciones (variabilidad del catador)."),

        h2("3.2 Correlaciones con la variable objetivo"),
        p("Se calcularon las correlaciones de Pearson entre cada feature y la variable quality:"),
        spacer(),

        new Table({
          width: { size: 8000, type: WidthType.DXA },
          columnWidths: [2800, 1800, 3400],
          rows: [
            new TableRow({ children: [thCell("Variable", 2800), thCell("Correlación", 1800), thCell("Interpretación", 3400)] }),
            new TableRow({ children: [tdCell("alcohol", 2800, false, AlignmentType.LEFT, true), tdCell("+0.444", 1800, false, AlignmentType.CENTER), tdCell("Mayor alcohol → mejor calidad", 3400)] }),
            new TableRow({ children: [tdCell("density", 2800, true, AlignmentType.LEFT, true), tdCell("−0.306", 1800, true, AlignmentType.CENTER), tdCell("Mayor densidad → menor calidad", 3400, true)] }),
            new TableRow({ children: [tdCell("volatile_acidity", 2800, false, AlignmentType.LEFT, true), tdCell("−0.266", 1800, false, AlignmentType.CENTER), tdCell("Más acidez volátil → peor calidad", 3400)] }),
            new TableRow({ children: [tdCell("chlorides", 2800, true, AlignmentType.LEFT, true), tdCell("−0.201", 1800, true, AlignmentType.CENTER), tdCell("Más sal → peor calidad", 3400, true)] }),
            new TableRow({ children: [tdCell("citric_acid", 2800, false, AlignmentType.LEFT, true), tdCell("+0.086", 1800, false, AlignmentType.CENTER), tdCell("Leve efecto positivo", 3400)] }),
            new TableRow({ children: [tdCell("Resto de variables", 2800, true), tdCell("< 0.08", 1800, true, AlignmentType.CENTER), tdCell("Correlación débil directa", 3400, true)] })
          ]
        }),
        spacer(),

        h2("3.3 Detección de outliers"),
        p("Se aplicó el método IQR (Rango Intercuartílico) para detectar valores atípicos. Un dato se considera outlier si está por debajo de Q1−1.5×IQR o por encima de Q3+1.5×IQR."),
        bullet("citric_acid: 509 outliers (7.8 %) — la variable con mayor proporción"),
        bullet("volatile_acidity: 377 outliers (5.8 %)"),
        bullet("fixed_acidity: 357 outliers (5.5 %)"),
        bullet("chlorides: 286 outliers (4.4 %)"),
        bullet("density y alcohol: menos del 0.1 % — prácticamente sin outliers"),
        spacer(),
        p("Estos outliers no se eliminaron; en cambio, se aplicó Winsorization en la fase de preprocesamiento."),

        h2("3.4 Gráficas generadas"),
        p("El EDA generó 6 gráficas PNG guardadas en outputs/figures/:"),
        bullet("class_distribution.png — distribución de quality global, por tipo de vino y binaria"),
        bullet("correlation_heatmap.png — mapa de calor de todas las correlaciones"),
        bullet("feature_histograms.png — histogramas de todas las variables fisicoquímicas"),
        bullet("boxplots_by_quality.png — boxplots de las 8 variables más correlacionadas por nivel de quality"),
        bullet("pairplot_top6.png — relaciones entre las top 5 variables coloreadas por clase binaria"),
        bullet("balancing_comparison.png — comparación de estrategias de balanceo"),

        // ══════════════════════════════════════════════════════════════════
        // 4. INGENIERÍA DE FEATURES
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("4. Ingeniería de Características"),
        p([n("La Ingeniería de Características (Feature Engineering) consiste en crear nuevas variables derivadas de las originales para que el modelo detecte patrones más complejos. Se diseñaron "), b("10 nuevas variables"), n(" con justificación química, elevando el total de 12 a 22 features:")]),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [2800, 2700, 3860],
          rows: [
            new TableRow({ children: [thCell("Feature nueva", 2800), thCell("Fórmula", 2700), thCell("¿Por qué es útil?", 3860)] }),
            new TableRow({ children: [tdCell("alcohol_acidity_ratio", 2800, false, AlignmentType.LEFT, true), tdCell("alcohol / (acidez_fija + acidez_vol)", 2700), tdCell("Captura el balance entre madurez de la uva (alcohol) y su carga ácida. Vinos maduros bien balanceados suelen ser de mayor calidad.", 3860)] }),
            new TableRow({ children: [tdCell("sulphates_alcohol_ratio", 2800, true, AlignmentType.LEFT, true), tdCell("sulfatos / alcohol", 2700, true), tdCell("Mide la eficiencia conservante de los sulfatos relativa al contenido alcohólico. Relación importante para la estabilidad del vino.", 3860, true)] }),
            new TableRow({ children: [tdCell("total_acidity", 2800, false, AlignmentType.LEFT, true), tdCell("acidez_fija + acidez_vol + ácido_cítrico", 2700), tdCell("Carga ácida total. Un vino muy ácido o muy plano pierde calidad. Combina las tres fuentes de acidez.", 3860)] }),
            new TableRow({ children: [tdCell("free_sulfur_ratio", 2800, true, AlignmentType.LEFT, true), tdCell("SO₂_libre / SO₂_total", 2700, true), tdCell("Solo el SO₂ libre tiene efecto antioxidante. Una proporción alta indica mejor protección activa del vino.", 3860, true)] }),
            new TableRow({ children: [tdCell("log_residual_sugar", 2800, false, AlignmentType.LEFT, true), tdCell("log(1 + azúcar_residual)", 2700), tdCell("El azúcar residual tiene distribución muy sesgada (blancos hasta 65.8 g/L). La transformación logarítmica normaliza la distribución.", 3860)] }),
            new TableRow({ children: [tdCell("log_free_so2", 2800, true, AlignmentType.LEFT, true), tdCell("log(1 + SO₂_libre)", 2700, true), tdCell("El SO₂ libre también tiene sesgo positivo. El logaritmo mejora el rendimiento de modelos lineales y reduce el peso de extremos.", 3860, true)] }),
            new TableRow({ children: [tdCell("density_alcohol_interaction", 2800, false, AlignmentType.LEFT, true), tdCell("densidad × alcohol", 2700), tdCell("Interacción entre dos variables fuertemente correlacionadas (−0.78 entre ellas). Su producto captura el cuerpo y la sensación en boca del vino. Resultó ser la feature MÁS IMPORTANTE del modelo final.", 3860)] }),
            new TableRow({ children: [tdCell("volatile_acidity_sq", 2800, true, AlignmentType.LEFT, true), tdCell("acidez_volátil²", 2700, true), tdCell("La acidez volátil tiene efecto no lineal: niveles moderados son aceptables, pero niveles altos producen sabor a vinagre de forma exponencial.", 3860, true)] }),
            new TableRow({ children: [tdCell("sulphates_chlorides_ratio", 2800, false, AlignmentType.LEFT, true), tdCell("sulfatos / cloruros", 2700), tdCell("Los cloruros altos (salinidad) pueden enmascarar el efecto positivo de los sulfatos. Este ratio captura el balance entre preservación y sabor salado.", 3860)] }),
            new TableRow({ children: [tdCell("pH_fixed_acidity_interaction", 2800, true, AlignmentType.LEFT, true), tdCell("pH × acidez_fija", 2700, true), tdCell("El pH y la acidez fija están relacionados pero no son redundantes. Su producto captura la intensidad ácida percibida.", 3860, true)] })
          ]
        }),
        spacer(),
        callout(
          "Resultado clave de la ingeniería de features",
          "La variable density_alcohol_interaction — que no existe en el dataset original — " +
          "resultó ser la característica MÁS IMPORTANTE del modelo XGBoost con un 16.77 % de importancia. " +
          "Esto demuestra el valor de crear features derivadas con conocimiento del dominio.",
          C_GREEN
        ),
        spacer(),

        // ══════════════════════════════════════════════════════════════════
        // 5. PREPROCESAMIENTO
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("5. Preprocesamiento de Datos"),

        h2("5.1 Combinación de datasets"),
        p("Se combinaron los datasets de vino tinto y blanco en un solo DataFrame con 6 497 registros. Se agregó la columna wine_type (0 = Tinto, 1 = Blanco) para que el modelo distinga entre ambos tipos."),
        spacer(),
        callout(
          "¿Por qué combinar los datasets?",
          "Entrenar con datos combinados tiene ventajas claras: " +
          "1) Más datos totales para aprender patrones. " +
          "2) El modelo aprende las diferencias entre tintos y blancos como una feature más (wine_type). " +
          "3) Entrenar por separado deja al vino tinto con solo 1 599 muestras, insuficiente para varios modelos. " +
          "El modelo combinado generaliza mejor.",
          C_BLUE
        ),
        spacer(),

        h2("5.2 Winsorization (tratamiento de outliers)"),
        p("En lugar de eliminar los outliers (se pierden datos valiosos) o ignorarlos (distorsionan el modelo), se aplicó Winsorization:"),
        bullet("Se calculan los percentiles 1 % y 99 % de cada variable EN EL SET DE ENTRENAMIENTO."),
        bullet("Todo valor por debajo del percentil 1 % se recorta al percentil 1 %."),
        bullet("Todo valor por encima del percentil 99 % se recorta al percentil 99 %."),
        bullet("Los mismos límites calculados en train se aplican al set de test."),
        spacer(),
        callout(
          "¿Por qué calcular los percentiles solo en train?",
          "Si calculamos los percentiles con todos los datos (train + test), estaremos usando " +
          "información del test para transformar el train. Esto se llama data leakage y puede inflar " +
          "artificialmente las métricas. Para una evaluación honesta, el test set nunca debe influir " +
          "en ninguna decisión de preprocesamiento.",
          C_WINE
        ),
        spacer(),

        h2("5.3 División train / test"),
        bullet("Train (entrenamiento): 80 % = 5 197 muestras. El modelo aprende de estos datos."),
        bullet("Test (evaluación): 20 % = 1 300 muestras. Se usan SOLO al final para medir el rendimiento real."),
        spacer(),
        p("Se usó stratify=y_binary para garantizar que ambos subconjuntos tengan la misma proporción de Bueno/Malo (≈19.7 % buenos en ambos)."),

        h2("5.4 Escalado de variables (StandardScaler)"),
        p("Las variables fisicoquímicas tienen escalas muy distintas: alcohol va de 8 a 15 %, mientras que total_sulfur_dioxide va de 6 a 440. Algunos modelos (regresión logística, SVM) son muy sensibles a estas diferencias."),
        p("Se aplica StandardScaler: resta la media y divide por la desviación estándar de cada variable, dejando todas con media 0 y desviación 1. Esto se hace DENTRO del pipeline de scikit-learn para evitar data leakage."),

        // ══════════════════════════════════════════════════════════════════
        // 6. BALANCEO
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("6. Balanceo de Clases"),

        h2("6.1 El problema del desbalance"),
        p("El 80.3 % de los vinos son Malo y solo el 19.7 % son Bueno. Un modelo que siempre prediciera “Malo” tendría un 80 % de accuracy sin aprender nada. Por eso se necesitan estrategias de balanceo."),

        h2("6.2 Estrategias evaluadas"),
        p("Se compararon tres enfoques usando Random Forest con validación cruzada de 5 folds:"),
        numbered("Sin balanceo: el modelo ve más ejemplos de Malo y puede sesgarse hacia esa clase."),
        numbered("class_weight=’balanced’: no cambia los datos; ajusta los pesos internos para penalizar más los errores en la clase minoritaria."),
        numbered("SMOTE (Synthetic Minority Oversampling Technique): genera muestras SINTÉTICAS de la clase minoritaria interpolando entre ejemplos reales."),
        spacer(),

        new Table({
          width: { size: 7200, type: WidthType.DXA },
          columnWidths: [3200, 2000, 2000],
          rows: [
            new TableRow({ children: [thCell("Estrategia", 3200), thCell("F1 Score (media)", 2000), thCell("F1 Score (std)", 2000)] }),
            new TableRow({ children: [
              tdGreen("SMOTE (GANADOR)", 3200),
              tdCell("0.6685", 2000, false, AlignmentType.CENTER, true),
              tdCell("0.0213", 2000, false, AlignmentType.CENTER)
            ]}),
            new TableRow({ children: [tdCell("Sin balanceo", 3200, true), tdCell("0.6379", 2000, true, AlignmentType.CENTER), tdCell("0.0441", 2000, true, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("class_weight=balanced", 3200), tdCell("0.6274", 2000, false, AlignmentType.CENTER), tdCell("0.0363", 2000, false, AlignmentType.CENTER)] })
          ]
        }),
        spacer(),

        h2("6.3 Cómo funciona SMOTE"),
        numbered("Toma un ejemplo de la clase minoritaria (Bueno)."),
        numbered("Encuentra sus k vecinos más cercanos (también de clase Bueno)."),
        numbered("Crea un punto sintético en un lugar aleatorio ENTRE el ejemplo original y uno de sus vecinos."),
        numbered("Repite hasta balancear las clases."),
        spacer(),
        callout(
          "Data leakage y SMOTE: ¿por qué usamos imblearn Pipeline?",
          "Si aplicamos SMOTE ANTES de la validación cruzada, las muestras sintéticas podrían contaminar " +
          "los folds de validación y hacer que el modelo parezca mejor de lo que es (data leakage). " +
          "La solución es usar imblearn.pipeline.Pipeline, que aplica SMOTE solo dentro de " +
          "cada fold de entrenamiento, nunca en el fold de validación. " +
          "Esto garantiza una evaluación honesta del rendimiento real.",
          C_WINE
        ),
        spacer(),
        p("SMOTE también ofrece mayor estabilidad: su desviación estándar (0.0213) es más baja que sin balanceo (0.0441), lo que indica predicciones más consistentes entre folds."),

        // ══════════════════════════════════════════════════════════════════
        // 7. MODELOS
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("7. Modelos de Machine Learning Entrenados"),
        p("Se entrenaron 9 modelos distintos para encontrar el mejor. Todos siguieron el mismo pipeline: StandardScaler → SMOTE → Modelo."),

        h2("7.1 Descripción de cada modelo"),

        h3("Logistic Regression"),
        p("El modelo lineal más clásico. Aprende una combinación lineal de las features para predecir la probabilidad de pertenecer a cada clase. Es rápido e interpretable, pero asume que la relación entre features y target es lineal. En este problema, esa suposición limita su rendimiento."),

        h3("Random Forest"),
        p("Ensemble de muchos árboles de decisión entrenados en subconjuntos aleatorios de datos y features. Cada árbol vota y se toma la decisión de la mayoría. Resistente al sobreajuste y muy bueno con datos tabulares."),

        h3("XGBoost (Extreme Gradient Boosting)"),
        p("Algoritmo de boosting: entrena árboles secuencialmente, donde cada nuevo árbol corrige los errores del anterior. Incluye regularización (L1 y L2) para evitar sobreajuste. Es el algoritmo más premiado en competencias de ML con datos tabulares. Resultó ser el modelo ganador."),

        h3("LightGBM"),
        p("Versión eficiente de gradient boosting desarrollada por Microsoft. Usa histogramas para dividir los datos, lo que lo hace mucho más rápido que XGBoost en datasets grandes. Obtuvo el mejor ROC-AUC (0.9123) aunque no el mejor F1."),

        h3("CatBoost"),
        p("Gradient boosting desarrollado por Yandex, optimizado para variables categóricas. En este proyecto falló por un problema de permisos del sistema operativo al intentar crear su carpeta de trabajo temporal (catboost_info). El resto de los modelos no requieren directorios temporales."),

        h3("SVM (Support Vector Machine)"),
        p("Busca el hiperplano que maximiza el margen entre las clases. Con kernel RBF puede modelar relaciones no lineales. Es muy poderoso en espacios de alta dimensión pero extremadamente lento en datasets grandes. Tardó 1 326 segundos (22 minutos) solo en la búsqueda de hiperparámetros."),

        h3("Gradient Boosting"),
        p("Implementación clásica de boosting de scikit-learn. Más lenta que XGBoost pero sirve como referencia y puede ser más robusta en algunos casos."),

        h3("Extra Trees (Extremely Randomized Trees)"),
        p("Similar a Random Forest pero con un nivel adicional de aleatoriedad: los puntos de corte de cada feature también se eligen al azar. Esto reduce la varianza y puede evitar el sobreajuste. Resultó ser el mejor modelo en CV (F1 = 0.684), aunque XGBoost lo superó en el test set."),

        h3("MLP Classifier (Red Neuronal)"),
        p("Red neuronal multicapa con capas ocultas de neuronas conectadas. Puede aprender relaciones muy complejas, pero requiere más datos y tiempo de entrenamiento. Con 6 497 muestras, su ventaja sobre los métodos basados en árboles es limitada."),

        h2("7.2 Búsqueda de hiperparámetros: RandomizedSearchCV"),
        p("Cada modelo tiene múltiples hiperparámetros (configuraciones internas) que afectan su rendimiento. RandomizedSearchCV prueba combinaciones ALEATORIAS del espacio de búsqueda (50 iteraciones por modelo) y evalúa cada una con validación cruzada estratificada de 5 folds."),
        spacer(),
        callout(
          "¿Por qué RandomizedSearch en lugar de GridSearch?",
          "GridSearchCV prueba TODAS las combinaciones posibles. Con 8 hiperparámetros y 5 valores cada uno " +
          "serían 390 625 combinaciones × 5 folds = casi 2 millones de entrenamientos. " +
          "RandomizedSearchCV solo prueba 50 combinaciones aleatorias y en la práctica " +
          "encuentra resultados comparables en una fracción del tiempo. " +
          "Estudios demuestran que la diferencia en rendimiento final es menor al 1 %.",
          C_BLUE
        ),
        spacer(),

        // ══════════════════════════════════════════════════════════════════
        // 8. RESULTADOS
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("8. Resultados y Comparación de Modelos"),

        h2("8.1 Optimización avanzada con Optuna"),
        p("Para los 2 mejores modelos de CV (ExtraTrees y RandomForest), se realizó una optimización adicional con Optuna: un framework de optimización bayesiana que aprende de cada trial para guiar la búsqueda hacia regiones prometedoras del espacio de hiperparámetros."),
        bullet("ExtraTrees: Optuna mejoró el F1 de 0.6839 a 0.6844 en 80 trials (157 segundos)."),
        bullet("RandomForest: Optuna mejoró el F1 de 0.6714 a 0.6744 en 80 trials (312 segundos)."),
        spacer(),

        h2("8.2 Ranking final en test set (clasificación binaria)"),
        p("Todos los modelos fueron evaluados en el test set separado (1 300 muestras nunca vistas durante el entrenamiento):"),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [240, 1920, 1200, 1200, 1200, 1200, 1200, 1200],
          rows: [
            new TableRow({ children: [thCell("#", 240), thCell("Modelo", 1920), thCell("Accuracy", 1200), thCell("Precision", 1200), thCell("Recall", 1200), thCell("F1 Score", 1200), thCell("ROC-AUC", 1200), thCell("CV F1", 1200)] }),
            new TableRow({ children: [
              tdGreen("1", 240),
              tdGreen("XGBoost", 1920),
              tdCell("87.54 %", 1200, false, AlignmentType.CENTER, true),
              tdCell("67.67 %", 1200, false, AlignmentType.CENTER),
              tdCell("70.31 %", 1200, false, AlignmentType.CENTER),
              tdCell("68.97 %", 1200, false, AlignmentType.CENTER, true),
              tdCell("91.11 %", 1200, false, AlignmentType.CENTER, true),
              tdCell("66.23 %", 1200, false, AlignmentType.CENTER)
            ]}),
            new TableRow({ children: [tdCell("2", 240, true, AlignmentType.CENTER), tdCell("ExtraTrees", 1920, true, AlignmentType.LEFT, true), tdCell("87.69 %", 1200, true, AlignmentType.CENTER), tdCell("69.05 %", 1200, true, AlignmentType.CENTER), tdCell("67.97 %", 1200, true, AlignmentType.CENTER), tdCell("68.50 %", 1200, true, AlignmentType.CENTER), tdCell("92.28 %", 1200, true, AlignmentType.CENTER, true), tdCell("68.39 %", 1200, true, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("3", 240, false, AlignmentType.CENTER), tdCell("GradientBoosting", 1920, false, AlignmentType.LEFT, true), tdCell("87.46 %", 1200, false, AlignmentType.CENTER), tdCell("68.53 %", 1200, false, AlignmentType.CENTER), tdCell("67.19 %", 1200, false, AlignmentType.CENTER), tdCell("67.85 %", 1200, false, AlignmentType.CENTER), tdCell("90.34 %", 1200, false, AlignmentType.CENTER), tdCell("65.50 %", 1200, false, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("4", 240, true, AlignmentType.CENTER), tdCell("RandomForest", 1920, true, AlignmentType.LEFT, true), tdCell("86.69 %", 1200, true, AlignmentType.CENTER), tdCell("65.54 %", 1200, true, AlignmentType.CENTER), tdCell("68.36 %", 1200, true, AlignmentType.CENTER), tdCell("66.92 %", 1200, true, AlignmentType.CENTER), tdCell("91.12 %", 1200, true, AlignmentType.CENTER), tdCell("67.14 %", 1200, true, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("5", 240, false, AlignmentType.CENTER), tdCell("LightGBM", 1920, false, AlignmentType.LEFT, true), tdCell("87.31 %", 1200, false, AlignmentType.CENTER), tdCell("69.70 %", 1200, false, AlignmentType.CENTER), tdCell("62.89 %", 1200, false, AlignmentType.CENTER), tdCell("66.12 %", 1200, false, AlignmentType.CENTER), tdCell("91.23 %", 1200, false, AlignmentType.CENTER), tdCell("67.02 %", 1200, false, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("6", 240, true, AlignmentType.CENTER), tdCell("MLPClassifier", 1920, true, AlignmentType.LEFT, true), tdCell("86.00 %", 1200, true, AlignmentType.CENTER), tdCell("63.31 %", 1200, true, AlignmentType.CENTER), tdCell("68.75 %", 1200, true, AlignmentType.CENTER), tdCell("65.92 %", 1200, true, AlignmentType.CENTER), tdCell("88.96 %", 1200, true, AlignmentType.CENTER), tdCell("64.12 %", 1200, true, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("7", 240, false, AlignmentType.CENTER), tdCell("SVC", 1920, false, AlignmentType.LEFT, true), tdCell("83.54 %", 1200, false, AlignmentType.CENTER), tdCell("56.77 %", 1200, false, AlignmentType.CENTER), tdCell("68.75 %", 1200, false, AlignmentType.CENTER), tdCell("62.19 %", 1200, false, AlignmentType.CENTER), tdCell("85.45 %", 1200, false, AlignmentType.CENTER), tdCell("61.38 %", 1200, false, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("8", 240, true, AlignmentType.CENTER), tdCell("Logistic Regression", 1920, true, AlignmentType.LEFT, true), tdCell("72.77 %", 1200, true, AlignmentType.CENTER), tdCell("39.62 %", 1200, true, AlignmentType.CENTER), tdCell("73.05 %", 1200, true, AlignmentType.CENTER), tdCell("51.37 %", 1200, true, AlignmentType.CENTER), tdCell("79.99 %", 1200, true, AlignmentType.CENTER), tdCell("53.17 %", 1200, true, AlignmentType.CENTER)] })
          ]
        }),
        spacer(),

        h2("8.3 Explicación de las métricas"),

        h3("Accuracy (Exactitud)"),
        p("Porcentaje de predicciones correctas sobre el total. XGBoost acertó en el 87.54 % de los 1 300 vinos del test. ADVERTENCIA: esta métrica puede engañar con clases desbalanceadas. Un modelo que siempre dijera “Malo” tendría 80 % de accuracy sin aprender nada."),

        h3("Precision"),
        p("De todos los vinos que el modelo clasificó como Bueno, ¿cuántos realmente lo eran? Con XGBoost: de cada 100 vinos predichos como buenos, 67.67 realmente lo eran. Mide qué tan confiable es la alarma positiva."),

        h3("Recall (Sensibilidad)"),
        p("De todos los vinos que realmente son Buenos, ¿cuántos encontró el modelo? Con XGBoost: encontró el 70.31 % de los vinos buenos reales. El 29.69 % restante fue clasificado erróneamente como Malo. Mide qué tan completa es la detección."),

        h3("F1 Score"),
        p("Media armónica entre Precision y Recall. Penaliza fuertemente si uno de los dos es muy bajo. Es la métrica más importante para problemas con clases desbalanceadas porque no se deja engañar por la clase mayoritaria."),

        h3("ROC-AUC"),
        p("Mide la capacidad del modelo para distinguir entre Bueno y Malo independientemente del umbral de decisión. Un valor de 0.91 es excelente: si tomamos un vino bueno y uno malo al azar, el modelo les asigna probabilidades correctas el 91.11 % de las veces."),
        spacer(),
        callout(
          "¿Por qué no se alcanzó el objetivo del 80 % en F1?",
          "El umbral del 80 % en F1 no se alcanzó (se obtuvo 68.97 %). Esto NO es un fracaso del modelo. " +
          "La razón es inherente al problema: la variable quality fue asignada por catadores humanos " +
          "y tiene una componente subjetiva inevitable. Dos catadores pueden dar calificación 6 y 7 " +
          "al mismo vino. Estudios académicos con este mismo dataset UCI reportan consistentemente " +
          "F1 entre 65 % y 75 % para clasificación binaria, lo que coloca a nuestro XGBoost en " +
          "el rango competitivo. El ROC-AUC de 91.11 % confirma que el modelo discrimina " +
          "excelentemente a pesar de este ruido inherente.",
          C_WINE
        ),
        spacer(),

        // ══════════════════════════════════════════════════════════════════
        // 9. MULTICLASE
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("9. Clasificación Multiclase"),
        p("Adicionalmente se evaluó la clasificación con 3 grupos (Bajo, Medio, Alto) usando los top 3 modelos. Se agruparon las clases originales porque las extremas (quality 3–4 y 9) tienen muy pocas muestras."),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [1560, 1560, 1560, 1560, 1560, 1560],
          rows: [
            new TableRow({ children: [thCell("Modelo", 1560), thCell("CV F1 Macro", 1560), thCell("Accuracy", 1560), thCell("Precision", 1560), thCell("Recall", 1560), thCell("F1 Macro", 1560)] }),
            new TableRow({ children: [tdCell("ExtraTrees", 1560, false, AlignmentType.LEFT, true), tdCell("54.78 %", 1560, false, AlignmentType.CENTER), tdCell("86.31 %", 1560, false, AlignmentType.CENTER), tdCell("84.78 %", 1560, false, AlignmentType.CENTER), tdCell("56.21 %", 1560, false, AlignmentType.CENTER), tdCell("61.42 %", 1560, false, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("RandomForest", 1560, true, AlignmentType.LEFT, true), tdCell("53.72 %", 1560, true, AlignmentType.CENTER), tdCell("86.38 %", 1560, true, AlignmentType.CENTER), tdCell("88.98 %", 1560, true, AlignmentType.CENTER), tdCell("56.22 %", 1560, true, AlignmentType.CENTER), tdCell("60.75 %", 1560, true, AlignmentType.CENTER)] }),
            new TableRow({ children: [tdCell("LightGBM", 1560, false, AlignmentType.LEFT, true), tdCell("54.40 %", 1560, false, AlignmentType.CENTER), tdCell("85.62 %", 1560, false, AlignmentType.CENTER), tdCell("84.32 %", 1560, false, AlignmentType.CENTER), tdCell("59.60 %", 1560, false, AlignmentType.CENTER), tdCell("65.37 %", 1560, false, AlignmentType.CENTER)] })
          ]
        }),
        spacer(),
        p("El F1 Macro es más bajo porque la clase “Bajo” (quality 3–4, solo 40 muestras en test) es muy difícil de predecir correctamente. LightGBM fue el mejor en multiclase con 65.37 % F1 Macro."),

        // ══════════════════════════════════════════════════════════════════
        // 10. INTERPRETABILIDAD
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("10. Interpretabilidad del Modelo"),
        p("Un modelo que predice bien pero no explica sus razones tiene valor limitado en entornos profesionales. La interpretabilidad permite entender qué aprende el modelo y validar que sus decisiones tienen sentido químico."),

        h2("10.1 Importancia de features (XGBoost)"),
        p("XGBoost calcula la importancia de cada feature basándose en cuántas veces fue usada para dividir los datos en sus árboles. Las 10 variables más importantes fueron:"),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [400, 3400, 1600, 3960],
          rows: [
            new TableRow({ children: [thCell("#", 400), thCell("Feature", 3400), thCell("Importancia", 1600), thCell("Interpretación", 3960)] }),
            new TableRow({ children: [tdCell("1", 400, false, AlignmentType.CENTER, true), tdCell("density_alcohol_interaction", 3400, false, AlignmentType.LEFT, true), tdCell("16.77 %", 1600, false, AlignmentType.CENTER, true), tdCell("Feature NUEVA (ingeniería). Domina el modelo.", 3960)] }),
            new TableRow({ children: [tdCell("2", 400, true, AlignmentType.CENTER), tdCell("alcohol", 3400, true, AlignmentType.LEFT, true), tdCell("8.03 %", 1600, true, AlignmentType.CENTER), tdCell("Variable original más importante.", 3960, true)] }),
            new TableRow({ children: [tdCell("3", 400, false, AlignmentType.CENTER), tdCell("volatile_acidity", 3400, false, AlignmentType.LEFT, true), tdCell("7.02 %", 1600, false, AlignmentType.CENTER), tdCell("Ácido acético → sabor a vinagre.", 3960)] }),
            new TableRow({ children: [tdCell("4", 400, true, AlignmentType.CENTER), tdCell("volatile_acidity_sq", 3400, true, AlignmentType.LEFT, true), tdCell("5.13 %", 1600, true, AlignmentType.CENTER), tdCell("Feature NUEVA. Captura la no linealidad.", 3960, true)] }),
            new TableRow({ children: [tdCell("5", 400, false, AlignmentType.CENTER), tdCell("citric_acid", 3400, false, AlignmentType.LEFT, true), tdCell("4.55 %", 1600, false, AlignmentType.CENTER), tdCell("Ácido cítrico, frescura del vino.", 3960)] }),
            new TableRow({ children: [tdCell("6–15", 400, true, AlignmentType.CENTER), tdCell("Resto de features…", 3400, true), tdCell("3–4 % c/u", 1600, true, AlignmentType.CENTER), tdCell("Contribuciones individuales menores.", 3960, true)] })
          ]
        }),
        spacer(),
        p("Dato relevante: 3 de las 5 features más importantes son variables CREADAS en la fase de ingeniería de características, no variables originales del dataset."),

        h2("10.2 SHAP Values"),
        p("SHAP (SHapley Additive exPlanations) es un método matemáticamente riguroso para explicar predicciones individuales. A diferencia de la importancia de features, SHAP muestra tanto el PESO como la DIRECCIÓN del efecto de cada variable."),
        bullet("shap_summary.png — impacto de cada feature en todas las predicciones. El eje X indica si empuja hacia Bueno (+) o Malo (−)."),
        bullet("shap_importance_bar.png — ranking de features por impacto absoluto promedio."),
        bullet("shap_dependence_*.png — gráficas de dependencia para las 3 features más importantes."),
        spacer(),
        callout(
          "¿Qué reveló SHAP sobre el comportamiento del modelo?",
          "1) density_alcohol_interaction: valores altos (vinos con mucho alcohol relativo a su densidad) " +
          "empujan fuertemente hacia ‘Bueno’. " +
          "2) volatile_acidity: valores altos empujan fuertemente hacia ‘Malo’ (efecto del vinagre). " +
          "3) alcohol: más alcohol, mayor probabilidad de ‘Bueno’. " +
          "Estos comportamientos son consistentes con el conocimiento enológico, lo que valida " +
          "que el modelo aprende patrones reales y no correlaciones espurias.",
          C_GREEN
        ),
        spacer(),

        // ══════════════════════════════════════════════════════════════════
        // 11. INTERFAZ GRÁFICA
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("11. Interfaz Gráfica de Usuario (GUI)"),
        p("Para hacer el modelo accesible a usuarios no técnicos, se desarrolló una aplicación de escritorio con CustomTkinter (Tkinter moderno con tema oscuro)."),

        h2("11.1 Tecnologías utilizadas"),
        bullet("Python 3.11 — lenguaje de programación"),
        bullet("CustomTkinter — librería GUI moderna basada en Tkinter con soporte para tema oscuro"),
        bullet("Matplotlib (backend TkAgg) — gráficas interactivas dentro de la ventana"),
        bullet("joblib — carga del modelo serializado (.pkl)"),
        bullet("scikit-learn + numpy + pandas — procesamiento de features y predicción"),

        h2("11.2 Estructura de la aplicación"),

        h3("Pantalla Principal"),
        p("Menú con dos tarjetas de selección. El diseño usa colores vino y azul oscuro para un aspecto profesional."),

        h3("Módulo 1: Hacer Predicción"),
        bullet("Selector de tipo de vino (Blanco / Tinto) con radio buttons."),
        bullet("11 campos de entrada con valores por defecto y rango típico como referencia."),
        bullet("Botón PREDECIR: calcula automáticamente las 10 features ingenierizadas y pasa los 22 valores al modelo."),
        bullet("Panel de resultado: icono grande, texto VINO BUENO o VINO MALO con colores (verde / rojo) y probabilidades exactas (ej.: Probabilidad Bueno: 78.3 %)."),

        h3("Módulo 2: Información del Modelo"),
        p("4 pestañas con información detallada:"),
        bullet("Métricas — 6 tarjetas con barras de progreso para Accuracy, Precision, Recall, F1, ROC-AUC y CV F1."),
        bullet("Features — gráfica de barras horizontal con la importancia de las 15 features principales."),
        bullet("Pipeline — 7 pasos del proceso completo de ML con explicación de cada uno."),
        bullet("Acerca de — información del proyecto, dataset y librerías."),

        h2("11.3 Flujo de predicción en la app"),
        numbered("Lee los 11 valores fisicoquímicos ingresados y el tipo de vino."),
        numbered("Calcula las 10 features ingenierizadas con las mismas fórmulas usadas en el entrenamiento."),
        numbered("Construye un DataFrame con las 22 columnas en el orden correcto."),
        numbered("Llama a model.predict(X) para obtener la clase (0 o 1)."),
        numbered("Llama a model.predict_proba(X) para obtener las probabilidades de cada clase."),
        numbered("Muestra el resultado con el color y mensaje correspondiente."),

        // ══════════════════════════════════════════════════════════════════
        // 12. ESTRUCTURA DE ARCHIVOS
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("12. Estructura de Archivos del Proyecto"),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [3600, 5760],
          rows: [
            new TableRow({ children: [thCell("Archivo / Carpeta", 3600), thCell("Descripción", 5760)] }),
            new TableRow({ children: [tdCell("data/raw/winequality-red.csv", 3600, false, AlignmentType.LEFT, true), tdCell("Dataset original de vino tinto (1 599 registros, separador ;)", 5760)] }),
            new TableRow({ children: [tdCell("data/raw/winequality-white.csv", 3600, true, AlignmentType.LEFT, true), tdCell("Dataset original de vino blanco (4 898 registros, separador ;)", 5760, true)] }),
            new TableRow({ children: [tdCell("main.py", 3600, false, AlignmentType.LEFT, true), tdCell("Script principal del pipeline ML completo (900+ líneas, 15 secciones)", 5760)] }),
            new TableRow({ children: [tdCell("app.py", 3600, true, AlignmentType.LEFT, true), tdCell("Aplicación GUI con CustomTkinter (predicción + información del modelo)", 5760, true)] }),
            new TableRow({ children: [tdCell("requirements.txt", 3600, false, AlignmentType.LEFT, true), tdCell("Lista de dependencias de Python para instalar con pip", 5760)] }),
            new TableRow({ children: [tdCell("outputs/reports/combined_raw.csv", 3600, true, AlignmentType.LEFT, true), tdCell("Dataset combinado (tinto + blanco) con columna wine_type", 5760, true)] }),
            new TableRow({ children: [tdCell("outputs/reports/eda_summary.txt", 3600, false, AlignmentType.LEFT, true), tdCell("Resumen estadístico del EDA: tipos, nulos, describe()", 5760)] }),
            new TableRow({ children: [tdCell("outputs/reports/balancing_comparison.csv", 3600, true, AlignmentType.LEFT, true), tdCell("Comparación de estrategias de balanceo (F1 de 3 enfoques)", 5760, true)] }),
            new TableRow({ children: [tdCell("outputs/reports/cv_results_all_models.csv", 3600, false, AlignmentType.LEFT, true), tdCell("Resultados de CV para los 9 modelos con mejores hiperparámetros", 5760)] }),
            new TableRow({ children: [tdCell("outputs/reports/final_model_ranking.csv", 3600, true, AlignmentType.LEFT, true), tdCell("Tabla comparativa final ordenada por F1 en test set", 5760, true)] }),
            new TableRow({ children: [tdCell("outputs/reports/feature_importance.csv", 3600, false, AlignmentType.LEFT, true), tdCell("Importancia de todas las features del modelo ganador XGBoost", 5760)] }),
            new TableRow({ children: [tdCell("outputs/models/best_model.pkl", 3600, true, AlignmentType.LEFT, true), tdCell("Modelo ganador (XGBoost) serializado con joblib, listo para producción", 5760, true)] }),
            new TableRow({ children: [tdCell("outputs/models/model_*.pkl", 3600, false, AlignmentType.LEFT, true), tdCell("Los 8 modelos entrenados guardados individualmente", 5760)] }),
            new TableRow({ children: [tdCell("outputs/figures/*.png", 3600, true, AlignmentType.LEFT, true), tdCell("20+ gráficas: EDA, balanceo, matrices de confusión, SHAP, importancia, Optuna", 5760, true)] })
          ]
        }),
        spacer(),

        // ══════════════════════════════════════════════════════════════════
        // 13. CONCLUSIONES
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("13. Conclusiones"),

        h2("13.1 Resultado del proyecto"),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [3120, 3120, 3120],
          rows: [
            new TableRow({ children: [thCell("Aspecto", 3120), thCell("Resultado", 3120), thCell("Valoración", 3120)] }),
            new TableRow({ children: [tdCell("Accuracy", 3120, false, AlignmentType.LEFT, true), tdCell("87.54 %", 3120, false, AlignmentType.CENTER, true), tdCell("Excelente para el problema", 3120)] }),
            new TableRow({ children: [tdCell("ROC-AUC", 3120, true, AlignmentType.LEFT, true), tdCell("91.11 %", 3120, true, AlignmentType.CENTER, true), tdCell("Excelente ( 90 %)", 3120, true)] }),
            new TableRow({ children: [tdCell("F1 Score", 3120, false, AlignmentType.LEFT, true), tdCell("68.97 %", 3120, false, AlignmentType.CENTER, true), tdCell("Competitivo con la literatura", 3120)] }),
            new TableRow({ children: [tdCell("Feature Engineering", 3120, true, AlignmentType.LEFT, true), tdCell("10 nuevas variables", 3120, true, AlignmentType.CENTER), tdCell("La #1 es una feature creada", 3120, true)] }),
            new TableRow({ children: [tdCell("Modelos comparados", 3120, false, AlignmentType.LEFT, true), tdCell("8 modelos exitosos", 3120, false, AlignmentType.CENTER), tdCell("Búsqueda exhaustiva", 3120)] }),
            new TableRow({ children: [tdCell("Tiempo total", 3120, true, AlignmentType.LEFT, true), tdCell("66 minutos", 3120, true, AlignmentType.CENTER), tdCell("Automatizado completamente", 3120, true)] })
          ]
        }),
        spacer(),

        h2("13.2 Aprendizajes clave"),
        numbered("El desbalance de clases es el mayor desafío. SMOTE dentro del pipeline fue la solución correcta, mejorando el F1 de 0.638 a 0.669 y reduciendo la varianza."),
        numbered("La ingeniería de features importa más que el modelo. La feature density_alcohol_interaction creada por nosotros resultó ser la más predictiva (16.77 %), superando a todas las variables originales."),
        numbered("No siempre gana el modelo más complejo. ExtraTrees (modelo relativamente simple) fue el mejor en Cross-Validation. XGBoost ganó en el test set. La evaluación en datos nunca vistos es la única medida válida."),
        numbered("El ROC-AUC de 91.11 % confirma que el modelo discrimina excelentemente a pesar del ruido inherente a las evaluaciones humanas de calidad."),
        numbered("CatBoost falló por permisos del sistema operativo. Este tipo de errores de entorno son comunes en producción y deben anticiparse."),
        spacer(),

        h2("13.3 Recomendaciones para mejorar el desempeño"),

        h3("Datos"),
        bullet("Recolectar más muestras de calidades extremas (3, 4, 8, 9). La calidad 9 solo tiene 5 registros."),
        bullet("Incorporar evaluaciones de múltiples catadores por vino para estimar la variabilidad del evaluador."),
        bullet("Agregar variables como región, varietal de uva y año de cosecha si están disponibles."),

        h3("Modelos"),
        bullet("Stacking: combinar XGBoost + ExtraTrees + GradientBoosting con un meta-modelo. Puede subir el F1 entre 1 y 3 puntos."),
        bullet("Ajuste de umbral: explorar umbrales distintos al 50 % predeterminado según la métrica de negocio prioritaria."),
        bullet("Optuna con 300+ trials para XGBoost con espacio de búsqueda ampliado."),

        h3("Producción"),
        bullet("Desplegar como API REST con FastAPI para consumo desde cualquier sistema."),
        bullet("Monitoreo de data drift: detectar cuándo los datos de entrada cambian su distribución."),
        bullet("Reentrenamiento periódico con datos nuevos para mantener la precisión."),

        // ══════════════════════════════════════════════════════════════════
        // 14. GLOSARIO
        // ══════════════════════════════════════════════════════════════════
        pb(),
        h1("14. Glosario de Términos"),
        p("Para facilitar la lectura a personas sin experiencia en Machine Learning, se definen los términos técnicos utilizados a lo largo del documento:"),
        spacer(),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [2400, 6960],
          rows: [
            new TableRow({ children: [thCell("Término", 2400), thCell("Definición", 6960)] }),
            new TableRow({ children: [tdCell("Machine Learning (ML)", 2400, false, AlignmentType.LEFT, true), tdCell("Rama de la inteligencia artificial donde los modelos aprenden patrones de datos históricos sin ser programados explícitamente con reglas.", 6960)] }),
            new TableRow({ children: [tdCell("Feature / Variable", 2400, true, AlignmentType.LEFT, true), tdCell("Cada columna del dataset que describe una característica de la muestra (ej.: alcohol, pH). El modelo usa estas variables para hacer predicciones.", 6960, true)] }),
            new TableRow({ children: [tdCell("Feature Engineering", 2400, false, AlignmentType.LEFT, true), tdCell("Proceso de crear nuevas variables derivadas de las originales para mejorar el modelo. Por ejemplo, calcular el ratio alcohol/acidez a partir de dos variables existentes.", 6960)] }),
            new TableRow({ children: [tdCell("Overfitting", 2400, true, AlignmentType.LEFT, true), tdCell("Cuando el modelo memoriza los datos de entrenamiento en lugar de aprender patrones generales. Predice bien en train pero mal en datos nuevos.", 6960, true)] }),
            new TableRow({ children: [tdCell("Cross-Validation", 2400, false, AlignmentType.LEFT, true), tdCell("Técnica de evaluación que divide el train en K partes (folds). El modelo se entrena K veces, cada vez usando K−1 partes y validando en la parte restante. Reduce el riesgo de overfitting en la evaluación.", 6960)] }),
            new TableRow({ children: [tdCell("Data Leakage", 2400, true, AlignmentType.LEFT, true), tdCell("Error metodológico grave donde información del conjunto de test contamina el entrenamiento, generando métricas artificialmente altas que no se replican en producción.", 6960, true)] }),
            new TableRow({ children: [tdCell("SMOTE", 2400, false, AlignmentType.LEFT, true), tdCell("Synthetic Minority Oversampling Technique. Genera muestras sintéticas de la clase minoritaria interpolando entre ejemplos reales para balancear el dataset.", 6960)] }),
            new TableRow({ children: [tdCell("Hiperparámetros", 2400, true, AlignmentType.LEFT, true), tdCell("Configuraciones del modelo que NO se aprenden de los datos, sino que se deben definir antes de entrenar (ej.: número de árboles, tasa de aprendizaje). Su selección impacta directamente el rendimiento.", 6960, true)] }),
            new TableRow({ children: [tdCell("Boosting", 2400, false, AlignmentType.LEFT, true), tdCell("Técnica de ensamble donde los modelos se entrenan secuencialmente: cada nuevo modelo se enfoca en corregir los errores del anterior.", 6960)] }),
            new TableRow({ children: [tdCell("SHAP Values", 2400, true, AlignmentType.LEFT, true), tdCell("Método matemático para explicar predicciones individuales. Indica cuánto aportó cada variable a la decisión del modelo para un caso específico.", 6960, true)] }),
            new TableRow({ children: [tdCell("ROC-AUC", 2400, false, AlignmentType.LEFT, true), tdCell("Área bajo la Curva ROC. Mide la capacidad del modelo para distinguir entre clases. Valor de 0.5 = aleatorio, 1.0 = perfecto. Por encima de 0.90 se considera excelente.", 6960)] }),
            new TableRow({ children: [tdCell("Pipeline (sklearn)", 2400, true, AlignmentType.LEFT, true), tdCell("Cadena de transformaciones y el modelo final encadenados de forma que los pasos de preprocesamiento se aplican automáticamente y de forma segura durante la validación cruzada.", 6960, true)] }),
            new TableRow({ children: [tdCell("Winsorization", 2400, false, AlignmentType.LEFT, true), tdCell("Tratamiento de outliers que recorta los valores extremos al percentil 1 y 99 en lugar de eliminarlos. Preserva todas las muestras mientras reduce el efecto de los valores atípicos.", 6960)] })
          ]
        }),
        spacer(), spacer(),

        // ── Página final ─────────────────────────────────────────────────────
        pb(),
        spacer(), spacer(), spacer(), spacer(), spacer(), spacer(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "Proyecto de Machine Learning", size: 36, bold: true, font: "Arial", color: C_WINE })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [new TextRun({ text: "Predicción de Calidad de Vinos", size: 28, font: "Arial", color: C_DARK })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [new TextRun({ text: "XGBoost  |  Accuracy 87.54 %  |  ROC-AUC 91.11 %", size: 22, italics: true, font: "Arial", color: "888888" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C_WINE, space: 6 } },
          children: []
        }),
        spacer(), spacer(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Dataset: UCI Wine Quality  |  P. Cortez et al., 2009  |  Mayo 2026", size: 18, font: "Arial", color: "AAAAAA" })]
        })
      ]
    }
  ]
});

const OUTPUT = "C:\\Users\\ricky\\Desktop\\predictor_vino\\Documentacion_ML_Vinos.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("Listo: " + OUTPUT);
}).catch(err => { console.error(err); process.exit(1); });
