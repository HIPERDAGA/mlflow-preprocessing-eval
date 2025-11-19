# -*- coding: utf-8 -*-
"""
Evalúa Multi-Scale Retinex (MSR) bajo una condición (ej. 'fog')
usando métricas NIQE/PIQE y registra resultados en MLflow.

Ejemplo de uso en Colab:

    from google.colab import drive
    drive.mount('/content/drive')

    !git clone https://github.com/HIPERDAGA/mlflow-preprocessing-eval.git
    %cd mlflow-preprocessing-eval

    !pip install -r requirements.txt

    import os
    os.environ["MLFLOW_TRACKING_URI"] = "file:///content/drive/MyDrive/mlruns"

    !python eval_msr.py \
        --data "/content/drive/MyDrive/datasets/fog" \
        --condition fog \
        --s1 15 --s2 80 --s3 250 \
        --gain 1.0 --offset 0.0 \
        --experiment "fog-preprocessing-evaluation"
"""

import os
import argparse
import numpy as np
import mlflow

from utils.dataset_loader import load_images_bgr, bgr_to_rgb
from preprocessing.msr_retinex import apply_msr_retinex
from metrics.niqe_metric import calculate_niqe
from metrics.piqe_metric import calculate_piqe

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="Carpeta con imágenes de la condición adversa (ej. fog)")
    ap.add_argument("--condition", default="fog",
                    help="Etiqueta de la condición (fog/rain/snow/lowlight)")

    ap.add_argument("--s1", type=float, default=15.0,
                    help="Sigma de la escala 1 para MSR")
    ap.add_argument("--s2", type=float, default=80.0,
                    help="Sigma de la escala 2 para MSR")
    ap.add_argument("--s3", type=float, default=250.0,
                    help="Sigma de la escala 3 para MSR")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="Ganancia lineal aplicada al resultado MSR")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="Desplazamiento lineal aplicado al resultado MSR")

    ap.add_argument("--experiment", default="fog-preprocessing-evaluation",
                    help="Nombre del experimento en MLflow")
    return ap.parse_args()

def main():
    args = parse_args()

    mlflow.set_experiment(args.experiment)

    run_name = f"MSR_{args.condition}"
    with mlflow.start_run(run_name=run_name):
        # Parámetros registrados
        mlflow.log_param("technique", "MSR")
        mlflow.log_param("condition", args.condition)
        mlflow.log_param("s1", args.s1)
        mlflow.log_param("s2", args.s2)
        mlflow.log_param("s3", args.s3)
        mlflow.log_param("gain", args.gain)
        mlflow.log_param("offset", args.offset)

        imgs_bgr = load_images_bgr(args.data)

        niqe_vals, piqe_vals = [], []

        for img_bgr in imgs_bgr:
            msr_bgr = apply_msr_retinex(
                img_bgr,
                s1=args.s1,
                s2=args.s2,
                s3=args.s3,
                gain=args.gain,
                offset=args.offset,
            )

            msr_rgb = bgr_to_rgb(msr_bgr)

            niqe_vals.append(calculate_niqe(msr_rgb))
            piqe_vals.append(calculate_piqe(msr_rgb))

        niqe_avg = float(np.mean(niqe_vals))
        piqe_avg = float(np.mean(piqe_vals))

        mlflow.log_metric("NIQE_avg", niqe_avg)
        mlflow.log_metric("PIQE_avg", piqe_avg)

        # Artefacto resumen
        os.makedirs("artifacts_msr", exist_ok=True)
        summary_path = os.path.join("artifacts_msr", "summary_msr.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(
                f"Technique: MSR\n"
                f"Condition: {args.condition}\n"
                f"s1: {args.s1}\n"
                f"s2: {args.s2}\n"
                f"s3: {args.s3}\n"
                f"gain: {args.gain}\n"
                f"offset: {args.offset}\n"
                f"NIQE_avg: {niqe_avg:.4f}\n"
                f"PIQE_avg: {piqe_avg:.4f}\n"
            )
        mlflow.log_artifact(summary_path)

        print("✅ Evaluación MSR completada.")
        print(f"NIQE_avg: {niqe_avg:.4f} | PIQE_avg: {piqe_avg:.4f}")

if __name__ == "__main__":
    main()
