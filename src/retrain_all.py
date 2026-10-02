"""Entrena todos los modelos predictivos mediante un único comando."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
TRAINERS = {
    "lv1_vw1": Path("src/level1/lv1_vw1_train.py"),
    "lv1_vw2": Path("src/level1/lv1_vw2_train.py"),
    "lv1_vw3": Path("src/level1/lv1_vw3_train.py"),
    "lv1_vw4": Path("src/level1/lv1_vw4_train.py"),
    "lv1_vw5": Path("src/level1/lv1_vw5_train.py"),
    "lv1_vw6": Path("src/level1/lv1_vw6_train.py"),
    "lv2_vw1": Path("src/level2/lv2_vw1_train.py"),
    "lv3_vw1": Path("src/level3/lv3_vw1_train.py"),
    "lv3_vw2": Path("src/level3/lv3_vw2_train.py"),
    "lv3_vw3": Path("src/level3/lv3_vw3_train.py"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(TRAINERS),
        help="Entrena solo los identificadores indicados.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Detiene el proceso al encontrar el primer entrenamiento fallido.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Guarda un resumen JSON en la ruta indicada.",
    )
    return parser.parse_args()


def read_metrics(model_key):
    metrics_path = MODELS_ROOT / f"train_{model_key}" / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    return {
        "selected_model": metrics.get("selected_model"),
        "variation_required": metrics.get("variation_required"),
        "available_models": metrics.get("available_models"),
        "train_years": metrics.get("train_years"),
        "test_years": metrics.get("test_years"),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
    }


def run_trainer(model_key):
    trainer_path = PROJECT_ROOT / TRAINERS[model_key]
    started = datetime.now().isoformat()
    process = subprocess.run(
        [sys.executable, str(trainer_path)],
        cwd=PROJECT_ROOT,
        text=True,
    )
    return {
        "model": model_key,
        "trainer": str(TRAINERS[model_key]),
        "started_at": started,
        "exit_code": process.returncode,
        "status": "ok" if process.returncode == 0 else "failed",
        "metrics": read_metrics(model_key),
    }


def main():
    args = parse_args()
    selected = args.only or list(TRAINERS)
    results = []

    print(f"Reentrenamiento iniciado: {len(selected)} modelos")
    for model_key in selected:
        print(f"\n=== {model_key} ===")
        result = run_trainer(model_key)
        results.append(result)
        print(f"Resultado: {result['status']} (exit_code={result['exit_code']})")
        if args.fail_fast and result["status"] == "failed":
            break

    summary = {
        "started_at": results[0]["started_at"] if results else None,
        "finished_at": datetime.now().isoformat(),
        "total": len(results),
        "successful": sum(result["status"] == "ok" for result in results),
        "failed": sum(result["status"] == "failed" for result in results),
        "results": results,
    }
    if args.json_output:
        output_path = args.json_output if args.json_output.is_absolute() else PROJECT_ROOT / args.json_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resumen JSON: {output_path}")

    print(
        f"\nFinalizado: {summary['successful']} correctos, "
        f"{summary['failed']} fallidos de {summary['total']}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


# .venv\Scripts\python.exe src\retrain_all.py //EJECUTAR TODOS
# .venv\Scripts\python.exe src\retrain_all.py --only lv1_vw1 lv1_vw2 // EJECUTAR SOLO LOS MODELOS INDICADOS
# .venv\Scripts\python.exe src\retrain_all.py --fail-fast // DETENER AL PRIMER ERROR
# .venv\Scripts\python.exe src\retrain_all.py --json-output reports\retrain_summary.json // GUARDAR RESUMEN JSON EN LA RUTA INDICADA
