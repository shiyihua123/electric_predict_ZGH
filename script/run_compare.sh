#!/bin/bash
# ==============================================================================
# 模型预测与 Montel 预测对比脚本（精简版）
# ==============================================================================

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 使用虚拟环境的 Python 路径
PYTHON="${PROJECT_DIR}/.venv/bin/python3"

# 对比脚本路径
COMPARE_SCRIPT="${PROJECT_DIR}/py_file/compare_predictions.py"

# 版本名称（与 run_train.sh 保持一致）
VERSION="my_new_experiment"

# ==============================================================================
# 必备参数配置
# ==============================================================================

# 模型预测结果文件（训练输出的 predictions_business.csv）
PRED_CSV="${PROJECT_DIR}/outputs/${VERSION}/models_results/predictions_business.csv"

# Montel 预测文件（当前只有 00 点发布的文件）
MONTEL_CSV="${PROJECT_DIR}/sourceData/SE2_Price_Spot_EUR_MWh_H_Forecast/forecast_issued_00_prefix_latest_fix.csv"

# 输出目录
OUT_DIR="${PROJECT_DIR}/outputs/${VERSION}/comparison_results"

# ==============================================================================
# 启动对比分析
# ==============================================================================

echo "========================================"
echo "  模型预测与 Montel 预测对比分析"
echo "========================================"
echo "模型预测文件: $PRED_CSV"
echo "Montel预测文件: $MONTEL_CSV"
echo "输出目录: $OUT_DIR"
echo "========================================"

# 检查文件是否存在
if [ ! -f "$PRED_CSV" ]; then
    echo "错误：模型预测文件不存在: $PRED_CSV"
    exit 1
fi

if [ ! -f "$MONTEL_CSV" ]; then
    echo "错误：Montel预测文件不存在: $MONTEL_CSV"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

# 运行对比脚本
"$PYTHON" "$COMPARE_SCRIPT" \
    --pred_csv "$PRED_CSV" \
    --montel_csv "$MONTEL_CSV" \
    --out_dir "$OUT_DIR"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  对比分析完成！"
    echo "========================================"
    echo "输出目录: $OUT_DIR"
else
    echo ""
    echo "========================================"
    echo "  对比分析失败！"
    echo "========================================"
    exit 1
fi
