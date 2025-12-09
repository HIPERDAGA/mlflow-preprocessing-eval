# -*- coding: utf-8 -*-
"""
Evaluación de la técnica de White Balance (Gray World) con NIQE / PIQE / BRISQUE.

Uso:
    python eval_white_balance.py \
        --data "ruta/a/imagenes" \
        --condition fog
"""

import argparse
import glob
import os

import cv2
import mlflow
import numpy as np

from preprocessing.white_balance import apply_white_balance
from metrics.niqe_metric import calculate_niqe
from metrics.piqe_metric import calculate_piqe
from metrics.brisque_metric import calculate_brisque  # el que añadimos antes


# --------- utilidades sencillas ---------

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def load_images_paths(folder: str):
    paths = []
    for ext in EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No se encontraron imágenes en {folder}")
    return paths


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# --------- evaluación ---------

def run_eval(args):
    mlflow.set_experiment("preprocessing_white_balance")

    img_paths = load_images_paths(args.data)

    niqe_vals, piqe_vals, brisque_vals = [], [], []

    with mlflow.start_run(run_name=f"white_balance_{args.condition}"):
        # Log de parámetros
        mlflow.log_param("technique", "white_balance_grayworld")
        mlflow.log_param("condition", args.condition)
        mlflow.log_param("wb_method", "grayworld")

        for p in img_paths:
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"⚠️ No se pudo leer la imagen: {p}")
                continue

            # --- preprocesado: white balance ---
            wb_bgr = apply_white_balance(img_bgr, method="grayworld")
            wb_rgb = bgr_to_rgb(wb_bgr)

            # --- métricas no-referenciales ---
            niqe_vals.append(calculate_niqe(wb_rgb))
            piqe_vals.append(calculate_piqe(wb_rgb))
            brisque_vals.append(calculate_brisque(wb_rgb))

        if not niqe_vals:
            raise RuntimeError("No se calcularon métricas (¿no hay imágenes válidas?).")

        niqe_avg = float(np.mean(niqe_vals))
        piqe_avg = float(np.mean(piqe_vals))
        brisque_avg = float(np.mean(brisque_vals))

        # Log en MLflow
        mlflow.log_metric("NIQE_avg", niqe_avg)
        mlflow.log_metric("PIQE_avg", piqe_avg)
        mlflow.log_metric("BRISQUE_avg", brisque_avg)

        # Resumen en texto
        summary = (
            f"Technique: WHITE_BALANCE_GRAYWORLD\n"
            f"Condition: {args.condition}\n"
            f"Images: {len(niqe_vals)}\n"
            f"NIQE_avg: {niqe_avg:.4f}\n"
            f"PIQE_avg: {piqe_avg:.4f}\n"
            f"BRISQUE_avg: {brisque_avg:.4f}\n"
        )

        print("✅ Evaluación White Balance completada.")
        print(summary)

        # Guardar resumen como artefacto de texto
        with open("summary_white_balance.txt", "w", encoding="utf-8") as f:
            f.write(summary)

        mlflow.log_artifact("summary_white_balance.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluar White Balance (Gray World) con NIQE/PIQE/BRISQUE"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Carpeta con las imágenes de entrada.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Condición (ej: fog, rain, snow, night, etc.)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
