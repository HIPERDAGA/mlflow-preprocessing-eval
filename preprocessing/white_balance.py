# -*- coding: utf-8 -*-
"""
preprocessing.white_balance

Módulo de preprocesado para corrección de balance de blancos (White Balance),
pensado para usarse dentro de los scripts de evaluación, de forma análoga a
`preprocessing.clahe_enhance`.

Trabajo en espacio BGR (OpenCV).
"""

import numpy as np
import cv2


def _grayworld_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica corrección de White Balance usando el método Gray World.

    Parámetros
    ----------
    img_bgr : np.ndarray
        Imagen de entrada en BGR uint8 con forma (H, W, 3).

    Returns
    -------
    np.ndarray
        Imagen BGR uint8 con corrección de balance de blancos.
    """
    if img_bgr is None:
        raise ValueError("img_bgr es None.")

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(
            f"Se esperaba imagen BGR con forma (H, W, 3) y se obtuvo {img_bgr.shape}"
        )

    img = img_bgr.astype(np.float32)

    # Medias por canal (BGR)
    b_mean, g_mean, r_mean = cv2.mean(img)[:3]
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    # Escalas por canal, evitando división por cero
    b_scale = gray_mean / (b_mean + 1e-6)
    g_scale = gray_mean / (g_mean + 1e-6)
    r_scale = gray_mean / (r_mean + 1e-6)

    img[:, :, 0] *= b_scale
    img[:, :, 1] *= g_scale
    img[:, :, 2] *= r_scale

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def apply_white_balance(img_bgr: np.ndarray, method: str = "grayworld") -> np.ndarray:
    """
    Función pública análoga a `apply_clahe` en clahe_enhance.py.

    Parámetros
    ----------
    img_bgr : np.ndarray
        Imagen BGR uint8 de entrada.
    method : str
        Método de WB. De momento solo 'grayworld'.

    Returns
    -------
    np.ndarray
        Imagen BGR uint8 procesada.
    """
    method = method.lower()

    if method == "grayworld":
        return _grayworld_white_balance(img_bgr)

    raise ValueError(f"Método de white balance no soportado: {method}")
