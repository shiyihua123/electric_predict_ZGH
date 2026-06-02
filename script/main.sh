#!/bin/bash
set -e

# 版本名称（与 run_train.sh 保持一致）
VERSION="my_new_experiment"

# ============================================================
# 条件判断参数
# ============================================================
bash "script/run_train.sh"
bash "script/run_compare.sh"
