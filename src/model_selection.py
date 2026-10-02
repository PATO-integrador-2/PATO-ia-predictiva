"""Selección de modelos temporales con control contra pronósticos constantes."""

import numpy as np


def predictions_vary(values, tolerance=1e-8):
    """Indica si un conjunto de predicciones contiene variación real."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return len(array) > 1 and float(np.ptp(array)) > tolerance


def select_best_model(predictions, metrics):
    """Elige el menor RMSE, prefiriendo candidatos que no sean constantes."""
    varying = [name for name, values in predictions.items() if predictions_vary(values)]
    candidates = varying or list(predictions)
    selected = min(candidates, key=lambda name: metrics[name]['rmse'])
    return selected, bool(varying)