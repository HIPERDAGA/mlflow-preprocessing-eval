# -*- coding: utf-8 -*-
"""
Corrección de balance de blancos (White Balance) con el método Gray World.

Trabaja en BGR (como OpenCV) y devuelve BGR.
"""

from typing import Literal
import numpy as np
import cv2


def apply_white_balance_grayworld(
    img_bgr: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """
    Aplica corrección de white balance con el supuesto Gray World.

    Parámetros
    ----------
    img_bgr : np.ndarray
        Imagen en BGR uint8.
    clip : bool
        Si True, recorta el resultado a [0,255].

    Returns
    -------
    np.ndarray
        Imagen BGR uint8 con corrección de balance de blancos.
    """
    if img_bgr is None:
        raise ValueError("img_bgr es None")

    img = img_bgr.astype(np.float32)

    # Medias por canal
    b_mean, g_mean, r_mean = cv2.mean(img)[:3]
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    # Evitar división por cero
    b_scale = gray_mean / (b_mean + 1e-6)
    g_scale = gray_mean / (g_mean + 1e-6)
    r_scale = gray_mean / (r_mean + 1e-6)

    img[:, :, 0] *= b_scale
    img[:, :, 1] *= g_scale
    img[:, :, 2] *= r_scale

    if clip:
        img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


def apply_white_balance(
    img_bgr: np.ndarray,
    method: Literal["grayworld"] = "grayworld",
    **kwargs,
) -> np.ndarray:
    """
    Wrapper por si luego quieres añadir otros métodos de WB.
    """
    if method == "grayworld":
        return apply_white_balance_grayworld(img_bgr, **kwargs)
    else:
        raise ValueError(f"Método de white balance no soportado: {method}")
