#!/bin/bash
# ==============================================================================
# 模型预测与 Montel 预测对比脚本（与真实电价比较）
# ==============================================================================
# 该脚本用于启动模型预测与 Montel 预测的对比分析，所有参数都在此脚本中配置。
#
# 发布时间点（瑞典时间）：
#   - 00:00 发布 → begin 是次日 00:00（target_start_days=1）
#   - 12:00 发布 → begin 是后日 00:00（target_start_days=2）
#
# 使用方法：
#   chmod +x run_compare.sh
#   ./run_compare.sh

# ==============================================================================
# 1. 基础配置
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 对比脚本路径
COMPARE_SCRIPT="./compare_predictions.py"

# ==============================================================================
# 2. 输入文件配置
# ==============================================================================

# 模型预测结果文件（训练输出的 predictions_business.csv）
PRED_CSV="./outputs/models_results/predictions_business.csv"

# Montel 预测文件
# 00点发布（瑞典时间）→ 预测次日0点开始
MONTEL_PREFIX_CSV="./sourceData/SE2_Price_Spot_EUR_MWh_H_Forecast/forecast_issued_00_prefix_latest.csv"
# 12点发布（瑞典时间）→ 预测后日0点开始
MONTEL_POSTFIX_CSV="./sourceData/SE2_Price_Spot_EUR_MWh_H_Forecast/forecast_issued_12_postfix_latest.csv"

# 发布时间点（只能是 0 或 12，表示 00:00 或 12:00 瑞典时间发布）
# 业务规则：
#   - 0 点发布 → begin 是次日 00:00（target_start_days=1）
#   - 12 点发布 → begin 是后日 00:00（target_start_days=2）
INSURED_TIME=0

# 真实电价数据文件
ACTUAL_CSV="./sourceData/SE2_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"

# 外部外生变量
EXOG_SE3_CSV=""
EXOG_SE1_CSV=""
EXOG_SE4_CSV=""
EXOG_SE2_WIND_CSV=""
EXOG_SE2_SOLAR_CSV=""

# ==============================================================================
# 3. 数据列配置（自动检测，一般不需要修改）
# ==============================================================================

# 日期列名（自动检测）
DATE_COL=""

# 价格列名（自动检测）
TARGET_COL=""

# ==============================================================================
# 4. 输出配置
# ==============================================================================

# 输出目录（会自动创建）
OUT_DIR="./outputs/comparison_results"

# ==============================================================================
# 5. CSV 文件配置
# ==============================================================================

# CSV 分隔符
CSV_SEP=","

# CSV 编码
CSV_ENCODING="utf-8"

# ==============================================================================
# 启动对比分析
# ==============================================================================

echo "========================================"
echo "  模型预测与 Montel 预测对比分析"
echo "  (与真实电价比较)"
echo "========================================"
echo "发布时间点（瑞典时间）："
echo "  - 0 点发布 → begin=次日00:00"
echo "  - 12 点发布 → begin=后日00:00"
echo "========================================"
echo "发布时间: $INSURED_TIME"
echo "模型预测文件: $PRED_CSV"
if [ "$INSURED_TIME" -eq 0 ]; then
    echo "Montel 预测文件: $MONTEL_PREFIX_CSV"
else
    echo "Montel 预测文件: $MONTEL_POSTFIX_CSV"
fi
echo "真实电价文件: $ACTUAL_CSV"
echo "输出目录: $OUT_DIR"
echo "========================================"

# 根据 insured_time 选择对应的 Montel 文件
if [ "$INSURED_TIME" -eq 0 ]; then
    MONTEL_CSV="$MONTEL_PREFIX_CSV"
else
    MONTEL_CSV="$MONTEL_POSTFIX_CSV"
fi

# 检查文件是否存在
if [ ! -f "$PRED_CSV" ]; then
    echo "错误：模型预测文件不存在: $PRED_CSV"
    exit 1
fi

if [ ! -f "$MONTEL_CSV" ]; then
    echo "错误：Montel预测文件不存在: $MONTEL_CSV"
    exit 1
fi

if [ ! -f "$ACTUAL_CSV" ]; then
    echo "错误：真实电价文件不存在: $ACTUAL_CSV"
    exit 1
fi

# 构建外生变量配置
# 用 Python 动态构建外生变量配置 JSON（仅添加文件存在的变量）
EXOG_CONFIGS=$(python3 -c "
import json, os
PROJECT_DIR = '$PROJECT_DIR'
configs = []
for path, name in [
    ('$EXOG_SE3_CSV', 'SE3'),
    ('$EXOG_SE1_CSV', 'SE1'),
    ('$EXOG_SE4_CSV', 'SE4'),
    ('$EXOG_SE2_WIND_CSV', 'SE2_WIND'),
    ('$EXOG_SE2_SOLAR_CSV', 'SE2_SOLAR'),
]:
    if not path:
        continue
    p = os.path.join(PROJECT_DIR, path) if not path.startswith('/') else path
    if os.path.isfile(p):
        configs.append({'csv_path': p, 'name': name})
print(json.dumps(configs))
")
echo "外部外生变量: $(echo "$EXOG_CONFIGS" | python3 -c "import json,sys; cf=json.load(sys.stdin); print([c['name'] for c in cf] if cf else '无')" 2>/dev/null)"

python3 "$COMPARE_SCRIPT" \
    --pred_csv "$PRED_CSV" \
    --montel_prefix_csv "$MONTEL_PREFIX_CSV" \
    --montel_postfix_csv "$MONTEL_POSTFIX_CSV" \
    --actual_csv "$ACTUAL_CSV" \
    --insured_time $INSURED_TIME \
    ${DATE_COL:+--date_col "$DATE_COL"} \
    ${TARGET_COL:+--target_col "$TARGET_COL"} \
    --out_dir "$OUT_DIR" \
    --csv_sep "$CSV_SEP" \
    --csv_encoding "$CSV_ENCODING" \
    --exog_configs "$EXOG_CONFIGS"

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
