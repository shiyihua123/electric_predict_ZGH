#!/bin/bash
# Plot forecast comparison: Montel forecast vs model predictions
#
# Usage:
#   bash run_plot.sh
#   bash run_plot.sh --model_name ours
#   bash run_plot.sh --model_name Ours --out_dir ./my_plots

set -euo pipefail

VENV_PYTHON="/home/syh/workplace/PythonProject/electric_predict_ZGH/.venv/bin/python3"

# ========== Defaults ==========
MONTEL_CSV="sourceData/SE2_Price_Spot_EUR_MWh_H_Forecast/forecast_issued_00_prefix_latest.csv"
PRED_CSV="outputs/predict_results/predict_$(date +%Y-%m-%d)_issued_00_1056h.csv"
MODEL_NAME="ours"
OUT_DIR="outputs/commits"

# ========== Parse CLI args ==========
print_help() {
    cat << EOF
Plot forecast comparison: Montel vs model predictions

Usage: bash run_plot.sh [options]

  Input:
    --montel_csv PATH          Montel forecast CSV file path
    --pred_csv PATH            Model predictions CSV file path

  Plot:
    --model_name NAME          Custom label for model in legend (default: original model name)
    --out_dir DIR              Output directory for saved plots (default: outputs_se3/plots)

Examples:
    bash run_plot.sh
    bash run_plot.sh --model_name ours
    bash run_plot.sh --model_name Ours --out_dir ./my_plots
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            print_help
            ;;
        --montel_csv)
            MONTEL_CSV="$2"; shift 2 ;;
        --pred_csv)
            PRED_CSV="$2"; shift 2 ;;
        --model_name)
            MODEL_NAME="$2"; shift 2 ;;
        --out_dir)
            OUT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# ========== Checks ==========
if [ ! -f "$MONTEL_CSV" ]; then
    echo "Error: Montel CSV not found: $MONTEL_CSV"
    exit 1
fi

if [ ! -f "$PRED_CSV" ]; then
    echo "Error: Model predictions CSV not found: $PRED_CSV"
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: venv Python not found: $VENV_PYTHON"
    exit 1
fi

# ========== Run ==========
echo "=========================================="
echo "  Plot Forecast Comparison"
echo "=========================================="
echo "Montel CSV  : $MONTEL_CSV"
echo "Pred CSV    : $PRED_CSV"
if [ -n "$MODEL_NAME" ]; then
    echo "Model label : $MODEL_NAME"
fi
echo "Output dir  : $OUT_DIR"
echo "=========================================="

PLOT_ARGS=(
    --montel_csv "$MONTEL_CSV"
    --pred_csv "$PRED_CSV"
    --out_dir "$OUT_DIR"
)
if [ -n "$MODEL_NAME" ]; then
    PLOT_ARGS+=(--model_name "$MODEL_NAME")
fi

"$VENV_PYTHON" "./plot_forecast.py" "${PLOT_ARGS[@]}"

echo ""
echo "=========================================="
echo "  Done. Plots saved to: $OUT_DIR"
echo "=========================================="
