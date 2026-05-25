#!/bin/bash
set -e

# ============================================================
# 条件判断参数
# ============================================================
bash "run_train.sh"
bash "run_predict.sh"
bash "run_compare.sh"
bash "run_plot.sh"
python "extract_predictions.py"
# python "daily_email.py"
