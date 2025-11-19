# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Sequence

def _msr_bgr(
    img_bgr: np.ndarray,
    scales: Sequence[float] = (15, 80, 250),
    weights: Sequence[float] | None = None,
    gain: float = 1.0,
    offset: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Multi-Scale Retinex clásico aplicado a imagen BGR.
    - img_bgr: imagen en BGR (uint8)
    - scales: sigmas de los filtros gaussianos (ej: [15, 80, 250])
    - weights: pesos para cada escala (si None -> uniformes)
    - gain, offset: ganancia y desplazamiento lineal
    - eps: pequeña constante para evitar log(0)
    Retorna: imagen BGR uint8 mejorada.
    """
    if img_bgr is None:
        raise ValueError("Imagen vacía o no cargada.")

    img = img_bgr.astype(np.float32) + 1.0  # evitar log(0)

    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)
    else:
        weights = np.array(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-6)

    retinex = np.zeros_like(img, dtype=np.float32)

    # Aplicar Retinex multi-escala canal por canal
    for sigma, w in zip(scales, weights):
        # GaussianBlur con kernel auto (0,0) y sigma definido
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        retinex += w * (np.log(img + eps) - np.log(blurred + eps))

    # Ganancia y desplazamiento
    retinex = gain * retinex + offset

    # Normalizar cada canal por separado a [0,255]
    out = np.zeros_like(retinex, dtype=np.float32)
    for c in range(3):
        ch = retinex[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min < 1e-6:
            out[:, :, c] = 0
        else:
            out[:, :, c] = (ch - ch_min) / (ch_max - ch_min) * 255.0

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def apply_msr_retinex(
    img_bgr: np.ndarray,
    s1: float = 15.0,
    s2: float = 80.0,
    s3: float = 250.0,
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Wrapper cómodo para MSR con 3 escalas.
    - s1, s2, s3: sigmas de las tres escalas gaussianas
    """
    scales = (s1, s2, s3)
    return _msr_bgr(img_bgr, scales=scales, gain=gain, offset=offset)
