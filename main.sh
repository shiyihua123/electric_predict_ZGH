#!/bin/bash
set -e

source "/home/syh/workplace/PythonProject/electric_predict_ZGH/.venv/bin/activate"
bash "run_train.sh"
bash "run_predict.sh"
bash "run_compare.sh"
bash "run_plot.sh"
python "extract_predictions.py"
# python "daily_email.py"
