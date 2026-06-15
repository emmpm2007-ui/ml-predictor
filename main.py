"""
main.py — Entry point for the ML Predictor System.

Supports three execution modes:

1. **Interactive** (no arguments): full guided wizard.
2. **Headless with CSV + config**: ``--data`` + ``--config`` flags.
3. **Load from saved model**: ``--load-model`` flag.

Usage examples::

    python main.py
    python main.py --data partidos.csv --config config.json --save-model modelo.pkl
    python main.py --load-model modelo.pkl --predict
    python main.py --data partidos.csv --config config.json --no-plots --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.predictor.config import Config
from src.predictor.data import DataManager
from src.predictor.persistence import ModelStore
from src.predictor.predictor import Predictor
from src.predictor.trainer import ModelTrainer
from src.predictor.ui import error, info, ok, titulo
from src.predictor.visualizer import Visualizador
from src.predictor.wizard import cargar_datos_interactivo, configurar_interactivo


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ml-predictor",
        description="Predictor universal de enfrentamientos deportivos con ML.",
    )
    parser.add_argument("--data",        metavar="CSV",   help="Ruta al CSV de entrenamiento.")
    parser.add_argument("--config",      metavar="JSON",  help="Ruta al archivo de configuración JSON.")
    parser.add_argument("--save-model",  metavar="PKL",   help="Guardar el modelo entrenado en este archivo.")
    parser.add_argument("--load-model",  metavar="PKL",   help="Cargar un modelo previamente guardado.")
    parser.add_argument("--predict",     action="store_true", help="Entrar en modo predicción tras cargar.")
    parser.add_argument("--no-plots",    action="store_true", help="Omitir la generación de gráficos.")
    parser.add_argument("--no-save",     action="store_true", help="No guardar la configuración automáticamente.")
    return parser.parse_args()


# ── Execution modes ────────────────────────────────────────────────────────────

def _mode_load(args: argparse.Namespace) -> None:
    """Load a pre-trained model and optionally run predictions."""
    cfg, trainer, data_mgr = ModelStore.cargar(args.load_model)
    if args.predict:
        pred = Predictor(cfg, trainer, data_mgr)
        pred.bucle_interactivo()


def _mode_train(args: argparse.Namespace) -> None:
    """Train from a CSV + JSON config (headless) or interactively."""
    # Config
    if args.config and Path(args.config).exists():
        cfg = Config.load(args.config)
        ok(f"Configuración cargada desde '{args.config}'.")
    else:
        cfg = configurar_interactivo()
        if not args.no_save:
            cfg.save("config_predictor.json")

    # Data
    data_mgr = DataManager(cfg)
    if args.data and Path(args.data).exists():
        df = data_mgr.cargar_csv(args.data)
    else:
        df = cargar_datos_interactivo(cfg)

    data_mgr.preparar(df)
    X      = data_mgr.get_X()
    y      = data_mgr.get_y_class()
    pesos  = data_mgr.get_pesos()
    y_A    = data_mgr.get_y_score(cfg.score_col_A) if cfg.score_col_A else None
    y_B    = data_mgr.get_y_score(cfg.score_col_B) if cfg.score_col_B else None

    # Training
    trainer = ModelTrainer(cfg)
    trainer.entrenar_clasificador(X, y, pesos)
    if y_A is not None or y_B is not None:
        trainer.entrenar_regresores(X, y_A, y_B)

    # Plots
    if not args.no_plots:
        viz = Visualizador(cfg, trainer)
        viz.graficar_todo(X, y, y_A, y_B)

    # Save model
    if args.save_model:
        ModelStore.guardar(args.save_model, cfg, trainer, data_mgr)
    elif not args.no_save:
        ModelStore.guardar("modelo_predictor.pkl", cfg, trainer, data_mgr)

    # Predictions
    pred = Predictor(cfg, trainer, data_mgr)
    pred.bucle_interactivo()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    titulo("🏆  ML PREDICTOR — SISTEMA DE PREDICCIÓN DEPORTIVA  🏆", 62)
    args = _parse_args()

    try:
        if args.load_model:
            _mode_load(args)
        else:
            _mode_train(args)
    except KeyboardInterrupt:
        print("\n")
        info("Sesión terminada por el usuario.")
        sys.exit(0)
    except Exception as exc:
        error(f"Error inesperado: {exc}")
        raise


if __name__ == "__main__":
    main()
