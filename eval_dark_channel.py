# -*- coding: utf-8 -*-
"""
Evalúa Dark Channel Prior (DCP) bajo una condición (ej. 'fog')
usando métricas NIQE/PIQE y registra resultados en MLflow.

Uso típico en Colab:

    from google.colab import drive
    drive.mount('/content/drive')

    !git clone https://github.com/HIPERDAGA/mlflow-preprocessing-eval.git
    %cd mlflow-preprocessing-eval

    !pip install -r requirements.txt

    import os
    os.environ["MLFLOW_TRACKING_URI"] = "file:///content/drive/MyDrive/mlruns"

    !python eval_dark_channel.py \
        --data "/content/drive/MyDrive/datasets/fog" \
        --condition fog \
        --win 15 --omega 0.95 --t0 0.1 \
        --experiment "fog-preprocessing-evaluation"
"""

import os
import argparse
import numpy as np
import mlflow

from utils.dataset_loader import load_images_bgr, bgr_to_rgb
from preprocessing.dark_channel_prior import apply_dark_channel_dehaze
from metrics.niqe_metric import calculate_niqe
from metrics.piqe_metric import calculate_piqe
from metrics.brisque_metric import calculate_brisque

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="Carpeta con imágenes de la condición adversa (ej. fog)")
    ap.add_argument("--condition", default="fog",
                    help="Etiqueta de la condición (fog/rain/snow/lowlight)")
    ap.add_argument("--win", type=int, default=15,
                    help="Tamaño de ventana para dark channel (impar típico 15)")
    ap.add_argument("--omega", type=float, default=0.95,
                    help="Factor de atenuación de haze (0.95 en el paper)")
    ap.add_argument("--t0", type=float, default=0.1,
                    help="Transmisión mínima para evitar saturación")
    ap.add_argument("--top_percent", type=float, default=0.001,
                    help="Fracción de píxeles para estimar la luz atmosférica A")
    ap.add_argument("--experiment", default="fog-preprocessing-evaluation",
                    help="Nombre del experimento en MLflow")
    return ap.parse_args()

def main():
    args = parse_args()

    mlflow.set_experiment(args.experiment)

    run_name = f"DCP_{args.condition}"
    with mlflow.start_run(run_name=run_name):
        # Log de parámetros
        mlflow.log_param("technique", "DarkChannelPrior")
        mlflow.log_param("condition", args.condition)
        mlflow.log_param("win_size", args.win)
        mlflow.log_param("omega", args.omega)
        mlflow.log_param("t0", args.t0)
        mlflow.log_param("top_percent", args.top_percent)

        imgs_bgr = load_images_bgr(args.data)
        niqe_vals, piqe_vals, brisque_vals = [], [], []
        
        for img_bgr in imgs_bgr:
            dehazed_bgr = apply_dark_channel_dehaze(
                img_bgr,
                win_size=args.win,
                omega=args.omega,
                t0=args.t0,
                top_percent=args.top_percent,
            )
            dehazed_rgb = bgr_to_rgb(dehazed_bgr)
        
            niqe_vals.append(calculate_niqe(dehazed_rgb))
            piqe_vals.append(calculate_piqe(dehazed_rgb))
            brisque_vals.append(calculate_brisque(dehazed_rgb))
        
        niqe_avg = float(np.mean(niqe_vals))
        piqe_avg = float(np.mean(piqe_vals))
        brisque_avg = float(np.mean(brisque_vals))
        
        mlflow.log_metric("NIQE_avg", niqe_avg)
        mlflow.log_metric("PIQE_avg", piqe_avg)
        mlflow.log_metric("BRISQUE_avg", brisque_avg)


        # Guardar resumen como artefacto
        os.makedirs("artifacts_dcp", exist_ok=True)
        summary_path = os.path.join("artifacts_dcp", "summary_dcp.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(
                f"Technique: DarkChannelPrior\n"
                f"Condition: {args.condition}\n"
                f"win_size: {args.win}\n"
                f"omega: {args.omega}\n"
                f"t0: {args.t0}\n"
                f"top_percent: {args.top_percent}\n"
                f"NIQE_avg: {niqe_avg:.4f}\n"
                f"PIQE_avg: {piqe_avg:.44f}\n"
            )
        mlflow.log_artifact(summary_path)

        print("✅ Evaluación DCP completada.")
        print(f"NIQE_avg: {niqe_avg:.4f} | PIQE_avg: {piqe_avg:.4f}")

if __name__ == "__main__":
    main()
