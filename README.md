## 👤 Autor

**Emanuelle Perez Martinez**

Proyecto desarrollado como parte de un portafolio personal en Python enfocado en Machine Learning aplicado, arquitectura modular y buenas prácticas de ingeniería de software.

---

# ML Predictor — Predictor Universal de Enfrentamientos Deportivos

Sistema profesional de predicción de resultados deportivos con múltiples modelos,
búsqueda de hiperparámetros, validación cruzada, feature engineering automático
y generación de reportes visuales.

![Versión Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Licencia MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-orange)
![sklearn](https://img.shields.io/badge/ML-scikit--learn-yellow)

---

## 🎯 Características

- **Multi-deporte**: fútbol (con empates), artes marciales, y cualquier deporte personalizado.
- **Multi-modelo**: RandomForest, GradientBoosting, MLP; XGBoost y LightGBM opcionales.
- **Búsqueda de hiperparámetros**: `RandomizedSearchCV` con `StratifiedKFold` para elegir automáticamente el mejor clasificador.
- **Feature engineering automático**: crea columnas de diferencia (`DIFF_`) y ratio (`RATIO_`) entre ambos competidores.
- **Regresores de puntuación**: predice el marcador numérico esperado de cada competidor.
- **Persistencia**: guarda y carga el modelo entrenado con `joblib` para no reentrenar.
- **Reportes visuales**: matriz de confusión, importancia de features, diagnóstico de regresores y SHAP (opcional).
- **Pesos temporales**: da más importancia a los enfrentamientos más recientes.
- **CLI completa**: modo interactivo guiado y modo headless con `--data` y `--config`.
- **Pruebas unitarias** para Config, DataManager, ModelTrainer y Predictor.

---

## 🚀 Instalación

### Requisitos

- Python 3.10 o superior.

### Clonar el repositorio

```bash
git clone https://github.com/emmpm2007-ui/ml-predictor.git
cd ml-predictor
```

### Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Módulos opcionales (mejoran la precisión)

```bash
pip install xgboost lightgbm shap
```

---

## ▶️ Uso

### Modo interactivo (guiado)

```bash
python main.py
```

El asistente te preguntará el deporte, nombres de los competidores, etiquetas de resultado
y características a usar. Ideal para empezar.

### Modo CLI con CSV y configuración

```bash
# Primera vez: entrena y guarda el modelo
python main.py --data partidos.csv --config config.json --save-model modelo.pkl

# Siguiente vez: carga el modelo y predice directamente
python main.py --load-model modelo.pkl --predict

# Sin gráficos (modo rápido)
python main.py --data partidos.csv --no-plots --no-save
```

### Modo automático

Edita `AUTO_CSV_PATH` y `AUTO_CONFIG_PATH` en `src/predictor/constants.py` para que el
script los use automáticamente al ejecutarse sin argumentos.

### Formato del CSV

El CSV debe tener columnas `A_<feat>` y `B_<feat>` para cada competidor, más una
columna de resultado:

```
A_goles_esperados_partido,A_posesion_pct,B_goles_esperados_partido,B_posesion_pct,resultado
1.8,55.2,1.1,44.8,A
0.9,48.1,1.6,51.9,B
1.2,50.0,1.2,50.0,E
```

---

## 🧪 Tests

```bash
pytest
pytest -v                         # salida detallada
pytest tests/test_data.py         # solo un módulo
```

Los tests usan datasets sintéticos en `tmp_path` y no requieren ningún CSV real.

---

## 🗂️ Estructura del proyecto

```
ml-predictor/
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── main.py                       ← punto de entrada CLI
├── docs/
│   └── TECHNICAL.md              ← arquitectura y decisiones de diseño
├── src/
│   └── predictor/
│       ├── __init__.py
│       ├── _compat.py            ← detección de XGBoost / LightGBM / SHAP
│       ├── constants.py          ← SUGERENCIAS, separadores, colores ANSI
│       ├── ui.py                 ← utilidades de terminal (color, pedir, confirmar)
│       ├── config.py             ← clase Config con serialización JSON
│       ├── data.py               ← DataManager: carga, preprocesado, feature engineering
│       ├── trainer.py            ← ModelTrainer: búsqueda, CV, entrenamiento
│       ├── visualizer.py         ← Visualizador: gráficos de diagnóstico
│       ├── predictor.py          ← Predictor: predicción interactiva
│       ├── persistence.py        ← ModelStore: guardar/cargar modelo
│       └── wizard.py             ← configurar_interactivo, cargar_datos_interactivo
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_data.py
    ├── test_trainer.py
    └── test_predictor.py
```

---

## 🤝 Contribuir

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para el flujo de trabajo y las guías de estilo.

---

## 📄 Licencia

MIT — ver [LICENSE](LICENSE).
