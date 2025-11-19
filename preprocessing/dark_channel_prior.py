# -*- coding: utf-8 -*-
import cv2
import numpy as np

def _get_dark_channel(img_rgb: np.ndarray, win_size: int = 15) -> np.ndarray:
    """
    Calcula el 'dark channel' de la imagen.
    img_rgb: imagen RGB uint8
    win_size: tamaño de ventana (impar), típico 15
    """
    # min en canales
    min_channel = np.min(img_rgb, axis=2)
    # min local (filtro min)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win_size, win_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def _estimate_atmospheric_light(img_rgb: np.ndarray,
                                dark: np.ndarray,
                                top_percent: float = 0.001) -> np.ndarray:
    """
    Estima la luz atmosférica A.
    top_percent: proporción de píxeles más brillantes del dark channel usados.
    """
    h, w = dark.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)

    # aplanar
    dark_vec = dark.reshape(-1)
    img_vec = img_rgb.reshape(-1, 3)

    # índices de los píxeles con dark más alto
    indices = np.argsort(dark_vec)[-num_top:]
    # tomar los correspondientes en la imagen original y escoger el más brillante
    brightest = img_vec[indices]
    A = brightest.mean(axis=0)  # promedio (también se puede usar max)
    return A

def _estimate_transmission(img_rgb: np.ndarray,
                           A: np.ndarray,
                           omega: float = 0.95,
                           win_size: int = 15) -> np.ndarray:
    """
    Estima el mapa de transmisión t(x).
    omega: factor de reducción (0.95 en el paper).
    """
    normed = img_rgb.astype(np.float32) / (A.reshape(1, 1, 3) + 1e-6)
    dark_norm = _get_dark_channel((normed * 255).astype(np.uint8), win_size)
    t = 1.0 - omega * (dark_norm.astype(np.float32) / 255.0)
    return t

def _recover_radiance(img_rgb: np.ndarray,
                      A: np.ndarray,
                      t: np.ndarray,
                      t0: float = 0.1) -> np.ndarray:
    """
    Recupera la imagen libre de niebla.
    t0: transmisión mínima para evitar división por cero.
    """
    t = np.clip(t, t0, 1.0)
    J = (img_rgb.astype(np.float32) - A.reshape(1, 1, 3)) / t[..., None] + A.reshape(1, 1, 3)
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

def apply_dark_channel_dehaze(img_bgr: np.ndarray,
                              win_size: int = 15,
                              omega: float = 0.95,
                              t0: float = 0.1,
                              top_percent: float = 0.001) -> np.ndarray:
    """
    Aplica dehazing basado en Dark Channel Prior.
    - img_bgr: imagen en BGR (cv2.imread)
    - win_size: ventana para dark channel
    - omega: peso del haze
    - t0: transmisión mínima
    - top_percent: fracción de píxeles para estimar A
    Retorna: imagen BGR dehazed.
    """
    if img_bgr is None:
        raise ValueError("Imagen vacía o no cargada.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    dark = _get_dark_channel(img_rgb, win_size=win_size)
    A = _estimate_atmospheric_light(img_rgb, dark, top_percent=top_percent)
    t = _estimate_transmission(img_rgb, A, omega=omega, win_size=win_size)
    J_rgb = _recover_radiance(img_rgb, A, t, t0=t0)

    img_bgr_out = cv2.cvtColor(J_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr_out
