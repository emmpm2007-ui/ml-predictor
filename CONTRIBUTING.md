# Contribuyendo a ML Predictor

¡Gracias por tu interés en contribuir! Este proyecto sigue un flujo de trabajo ligero:

## Cómo contribuir

1. Haz un fork del repositorio.
2. Crea una rama: `git checkout -b feature/nombre-mejora`.
3. Añade pruebas unitarias si el cambio modifica la lógica.
4. Ejecuta `black src tests` y `flake8 src tests` para mantener el estilo.
5. Asegúrate de que todas las pruebas pasan: `pytest`.
6. Haz commit con mensajes claros.
7. Abre un Pull Request describiendo los cambios.

## Estilo de código

- Seguimos PEP 8 con las reglas definidas en `pyproject.toml`.
- Usa docstrings en todas las funciones públicas (formato Google).
- Nombres de funciones, módulos y clases en inglés; mensajes de UI al usuario en español.
- Los modelos entrenados (`.pkl`) no se incluyen en el repositorio; solo el código para generarlos.

## Añadir un nuevo modelo

1. En `src/predictor/trainer.py`, agrega una entrada en `_clasificadores_candidatos()` o `_regresores_candidatos()`.
2. Sigue el patrón `{nombre: (modelo, param_grid)}` existente.
3. Añade la importación condicional en `src/predictor/_compat.py` si el modelo es opcional.
4. Escribe un test en `tests/test_trainer.py` que verifique que el candidato aparece en el diccionario.

## Añadir un nuevo deporte

1. Añade una clave con sus features sugeridas en `SUGERENCIAS` en `src/predictor/constants.py`.
2. La configuración `con_empate` y las etiquetas de resultado ya son genéricas.

## Reporte de problemas

Abre un Issue con:
- Descripción detallada.
- Pasos para reproducirlo.
- Versión de Python, sistema operativo y `pip freeze`.
