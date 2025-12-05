# -*- coding: utf-8 -*-
"""
Cálculo de BRISQUE usando pyiqa.

Requiere:
    pyiqa>=0.1.11
"""

from typing import Optional

import numpy as np
import torch
import pyiqa


# Inicializamos el métrico una sola vez para no recrearlo en cada llamada
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BRISQUE_METRIC: Optional[torch.nn.Module] = None


def _get_brisque_metric() -> torch.nn.Module:
    """Crea (si hace falta) y devuelve el métrico BRISQUE de pyiqa."""
    global _BRISQUE_METRIC
    if _BRISQUE_METRIC is None:
        # pyiqa.create_metric('brisque') carga el modelo y pesos por defecto
        _BRISQUE_METRIC = pyiqa.create_metric("brisque", device=_DEVICE)
        _BRISQUE_METRIC.eval()
    return _BRISQUE_METRIC


def calculate_brisque(img_rgb: np.ndarray) -> float:
    """
    Calcula BRISQUE para una imagen en RGB (H, W, 3), rango [0, 255].

    Parámetros
    ----------
    img_rgb : np.ndarray
        Imagen RGB uint8 o float, en formato HWC.

    Devuelve
    --------
    float
        Score BRISQUE (cuanto más bajo, mejor calidad percibida).
    """
    if img_rgb is None:
        raise ValueError("Imagen es None.")
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"Se esperaba imagen RGB con forma (H, W, 3), recibido {img_rgb.shape}"
        )

    # Aseguramos tipo y rango
    img = img_rgb.astype(np.float32)
    # Si viene en [0, 255], normalizamos a [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    # Convertir a tensor: (1, 3, H, W) en [0,1]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(_DEVICE)

    metric = _get_brisque_metric()
    with torch.no_grad():
        score = metric(tensor)

    # score es un tensor escalar
    return float(score.squeeze().cpu().item())
