# -*- coding: utf-8 -*-
"""eval_white_balance

Evalúa White Balance (Gray World) bajo una condición (p. ej., 'fog')
usando métricas NIQE/PIQE/BRISQUE y registra resultados en MLflow.

Uso (Colab):
    !pip install -r requirements.txt
    from google.colab import drive; drive.mount('/content/drive')
    import os; os.environ["MLFLOW_TRACKING_URI"] = "file:///content/drive/MyDrive/mlruns"
    !python eval_white_balance.py --data "/content/drive/MyDrive/datasets/fog" --condition fog
"""

import os
import argparse
import numpy as np
import mlflow

from utils.dataset_loader import load_images_bgr, bgr_to_rgb
from preprocessing.white_balance import apply_white_balance
from metrics.niqe_metric import calculate_niqe
from metrics.piqe_metric import calculate_piqe
from metrics.brisque_metric import calculate_brisque


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",required=True,
        help="Carpeta con imágenes de la condición adversa (ej. fog)")
    ap.add_argument("--condition",default="fog",
        help="Etiqueta de la condición (fog/rain/snow/lowlight)")
    
    ap.add_argument("--experiment",default="preprocessing-eval",
                    help="Nombre del experimento en MLflow")
    ap.add_argument("--method",default="grayworld",
                    help="Método de white balance (por ahora solo 'grayworld')")
    return ap.parse_args()


def main():
    args = parse_args()

    # Asegurar experimento
    mlflow.set_experiment(args.experiment)

    run_name = f"WB_{args.condition}"
    with mlflow.start_run(run_name=run_name):
        # Parámetros registrados
        mlflow.log_param("technique", "WHITE_BALANCE")
        mlflow.log_param("condition", args.condition)
        mlflow.log_param("wb_method", args.method)

        imgs_bgr = load_images_bgr(args.data)
        niqe_vals, piqe_vals, brisque_vals = [], [], []

        for img_bgr in imgs_bgr:
            # Preprocesamiento: WHITE BALANCE
            proc_bgr = apply_white_balance(
                img_bgr,
                method=args.method,
            )
            proc_rgb = bgr_to_rgb(proc_bgr)

            # Métricas no-referenciales
            niqe_vals.append(calculate_niqe(proc_rgb))
            piqe_vals.append(calculate_piqe(proc_rgb))
            brisque_vals.append(calculate_brisque(proc_rgb))

        # Promedios
        niqe_avg = float(np.mean(niqe_vals))
        piqe_avg = float(np.mean(piqe_vals))
        brisque_avg = float(np.mean(brisque_vals))

        mlflow.log_metric("NIQE_avg", niqe_avg)
        mlflow.log_metric("PIQE_avg", piqe_avg)
        mlflow.log_metric("BRISQUE_avg", brisque_avg)

        # Artefacto simple: guardamos un .txt con resumen
        os.makedirs("artifacts_wb", exist_ok=True) 
        summary_path = os.path.join("artifacts_wb", "summary_wb.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(
            f"Technique: WHITE_BALANCE\n"
            f"Condition: {args.condition}\n"
            f"method: {args.method}\n"
            f"NIQE_avg: {niqe_avg:.4f}\n"
            f"PIQE_avg: {piqe_avg:.4f}\n"
            f"BRISQUE_avg: {brisque_avg:.4f}\n"
        )
        mlflow.log_artifact(summary_path)

        print("✅ Evaluación WB completada.")
        print(f"NIQE_avg: {niqe_avg:.4f} | PIQE_avg: {piqe_avg:.4f} | BRISQUE_avg: {bisque_avg:.4f}")

if __name__ == "__main__":
    main()
