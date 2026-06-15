# Documentación Técnica – ML Predictor

## Arquitectura general

El proyecto sigue una **arquitectura de 10 módulos** con separación estricta de responsabilidades:

```
constants.py  _compat.py
     ↓              ↓
   ui.py         (xgboost/lightgbm/shap flags)
     ↓
  config.py
     ↓
  data.py ──────────────────────────────┐
     ↓                                  │
 trainer.py ──── visualizer.py          │
     ↓                                  │
 predictor.py ──────────────────────────┤
     ↓                                  │
persistence.py      wizard.py ──────────┘
     ↓                 ↓
              main.py (entry point)
```

Sin dependencias circulares. Cada módulo solo importa hacia "arriba" del grafo.

## Módulos

- **`constants.py`**: features sugeridas por deporte, separadores CSV, códigos ANSI, rutas en modo automático.
- **`_compat.py`**: detección de XGBoost, LightGBM y SHAP con graceful fallback. Expone `XGBOOST_OK`, `LIGHTGBM_OK`, `SHAP_OK` y las clases (`None` si no instaladas).
- **`ui.py`**: funciones de terminal: `c()` para colorear, `titulo()`, `info()`, `ok()`, `warn()`, `error()`, `pedir()`, `confirmar()`, `separador()`.
- **`config.py`**: `Config` — dataclass de configuración completa. Soporta `to_dict()` / `from_dict()` para serialización JSON y `save()` / `load()` para persistencia en disco.
- **`data.py`**: `DataManager` — carga CSV (auto-detecta separador), entrada manual por teclado, preprocesado (imputación mediana, feature engineering) y construcción del vector de predicción.
- **`trainer.py`**: `ModelTrainer` — selecciona el mejor clasificador con `RandomizedSearchCV` + `StratifiedKFold`, reporta métricas de CV, entrena regresores de puntuación y emite consejos personalizados.
- **`visualizer.py`**: `Visualizador` — genera y guarda en `reporte_predictor/`: matriz de confusión (raw + normalizada), importancia de features (top-20), diagnóstico de regresores (real vs predicho, residuos) y SHAP summary plot.
- **`predictor.py`**: `Predictor` — construye el vector de predicción, llama al clasificador y regresores, muestra probabilidades con barra visual y bucle interactivo.
- **`persistence.py`**: `ModelStore` — empaqueta `Config`, `ModelTrainer` y `DataManager` en un bundle `joblib` para guardar y restaurar el estado completo sin reentrenar.
- **`wizard.py`**: `configurar_interactivo()`, `cargar_datos_interactivo()`, `_verificar_columnas_csv()` — asistente guiado por consola para primera ejecución sin argumentos.

## Pipeline de datos

```
CSV / teclado
    → DataManager.cargar_csv() / cargar_manual()
    → DataManager.preparar():
        1. Validar columnas A_feat / B_feat
        2. Imputar nulos con mediana (SimpleImputer)
        3. Feature engineering: DIFF_feat = A_feat - B_feat
                                RATIO_feat = A_feat / B_feat
        4. Construir lista final de feature_cols
    → get_X() → pd.DataFrame (features)
    → get_y_class() → pd.Series (0=B gana, 1=A gana, 2=empate)
    → get_pesos() → np.ndarray | None (pesos temporales lineales)
```

## Pipeline de entrenamiento

```
ModelTrainer.entrenar_clasificador(X, y):
    Para cada candidato {RF, GB, MLP, [XGB, LGBM]}:
        RandomizedSearchCV(n_iter, StratifiedKFold(cv_folds))
        scoring = "f1_macro"
    Selecciona best_estimator_ con mayor F1-macro CV
    cross_validate() → accuracy + f1_macro (train + val)
    Entrenamiento final en todo X
    _consejos_precision() → recomendaciones personalizadas

ModelTrainer.entrenar_regresores(X, y_A, y_B):
    Para cada puntuación disponible:
        RandomizedSearchCV(scoring="neg_mean_squared_error")
        cross_validate() → R² + MAE
```

## Feature engineering

Los features `DIFF_` y `RATIO_` son las fuentes de señal más potentes para el clasificador porque capturan la **ventaja relativa** entre los competidores en cada dimensión. Para N features comunes entre A y B, se generan 2N features derivadas.

Los ratios se protegen frente a división por cero reemplazando `B_feat == 0` por `NaN` y luego imputando a `0`.

## Persistencia

`ModelStore.guardar()` serializa con `joblib.dump()` un bundle dict que incluye:
- `cfg`: dict de `Config` (JSON-serializable)
- `best_clf`, `best_reg_A`, `best_reg_B`: modelos sklearn
- `feature_cols`, `label_names`, `class_values`: metadatos de entrenamiento
- `dm_feature_cols`: columnas de features del DataManager
- `clf_report`, `reg_report`: métricas de validación cruzada

Al cargar, `DataManager.df` se inicializa con un DataFrame vacío (solo columnas), suficiente para `construir_vector_prediccion()` en modo predicción pura.

## Módulos opcionales

| Módulo | Beneficio | Instalación |
|---|---|---|
| XGBoost | Clasificador y regresor más potente | `pip install xgboost` |
| LightGBM | Velocidad en datasets grandes | `pip install lightgbm` |
| SHAP | Interpretabilidad de cada predicción | `pip install shap` |

Si no están instalados, el sistema funciona con RF, GB y MLP.

## Próximos pasos

- Añadir `StackingClassifier` para combinar los mejores modelos.
- Interfaz web con FastAPI.
- Escalar features para MLP con `Pipeline + StandardScaler` (actualmente sin escalado).
- Historial head-to-head como feature adicional.
- Exportar el reporte completo a PDF.
