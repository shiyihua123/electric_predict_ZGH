#!/bin/bash
set -e

# 版本名称（与 run_train.sh 保持一致）
VERSION="my_new_experiment"

# ============================================================
# 条件判断参数
# ============================================================
bash "run_train.sh"
bash "run_predict.sh"
bash "run_compare.sh"
bash "run_plot.sh"
python "extract_predictions.py" \
    --input "outputs/${VERSION}/predict_results/predict_$(date +%Y-%m-%d)_issued_00_1056h.csv" \
    --out_dir "outputs/${VERSION}/commits"
# python "daily_email.py"
