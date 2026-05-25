#!/bin/bash
# ==============================================================================
# 电价预测模型训练启动脚本
# ==============================================================================
# 该脚本用于启动电价预测模型的训练，所有参数都在此脚本中配置。
#
# 发布时间点（瑞典时间）：
#   - 00:00 发布 → begin 是次日 00:00（target_start_days=1）
#   - 12:00 发布 → begin 是后日 00:00（target_start_days=2）
#
# 使用方法：
#   chmod +x run_train.sh
#   ./run_train.sh

# ==============================================================================
# 1. 基础配置
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 训练脚本路径
TRAIN_SCRIPT="main.py"

# 输出目录（会自动创建）
OUT_DIR="./outputs/models_results"

# ==============================================================================
# 2. 数据配置
# ==============================================================================

# 数据文件路径
DATA_PATH="sourceData/SE2_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"

# 列名配置
DATE_COL="date"           # 时间列名
TARGET_COL="price"        # 目标列名（价格）
UNIQUE_ID="SE2"           # 时间序列标识

# 文件格式配置
CSV_SEP=","               # CSV 分隔符
CSV_ENCODING="utf-8"      # CSV 编码

# 缺失值处理策略：interpolate（插值）、ffill（前向填充）、raise（报错）
MISSING_STRATEGY="interpolate"

# ==============================================================================
# 3. 时间切分配置（UTC 时间）
# ==============================================================================

TRAIN_START="2015-01-01 00:00:00+00:00"
TRAIN_END="2023-12-31 23:00:00+00:00"

VAL_START="2024-01-01 00:00:00+00:00"
VAL_END="2024-12-31 23:00:00+00:00"

TEST_START="2025-01-01 00:00:00+00:00"
TEST_END="2026-05-18 23:00:00+00:00"

# ==============================================================================
# 4. 业务预测任务配置
# ==============================================================================

# 模型输入历史天数（建议 28 或 56）
INPUT_DAYS=90

# 发布时区（瑞典时区）
ISSUED_TZ="Europe/Stockholm"

# 发布时间点（只能是 0 或 12，表示 00:00 或 12:00 瑞典时间发布）
# 业务规则：
#   - 0 点发布 → begin 是次日 00:00（target_start_days=1）
#   - 12 点发布 → begin 是后日 00:00（target_start_days=2）
INSURED_TIME=0

# 预测小时数（5天 = 120小时）
TARGET_HOURS=1056

# 外部外生变量
EXOG_SE3_CSV="sourceData/SE3_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"
EXOG_SE1_CSV="sourceData/SE1_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"
EXOG_SE4_CSV="sourceData/SE4_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"
EXOG_SE2_WIND_CSV="sourceData/SE2_Wind_Power_Production_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_SOLAR_CSV="sourceData/SE2_Solar_Photovoltaic_Production_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_CONSUMPTION_CSV="sourceData/SE2_Consumption_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_HYDRO_CSV="sourceData/SE2_Hydro_Power_Production_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_RESIDUAL_LOAD_CSV="sourceData/SE2_Residual_Load_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_SE3_NTC_CSV="sourceData/SE2_SE3_Exchange_Net_Transfer_Capacity_MW_15min_REMIT/MW_latest.csv"
EXOG_SE2_SE3_FLOW_CSV="sourceData/SE2_SE3_Exchange_Physical_Flow_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE3_SE2_NTC_CSV="sourceData/SE3_SE2_Exchange_Net_Transfer_Capacity_MW_15min_REMIT/MW_latest.csv"
EXOG_SE3_SE2_FLOW_CSV="sourceData/SE3_SE2_Exchange_Physical_Flow_MWh_h_H_Actual/MW_latest.csv"
# ==============================================================================
# 5. 模型配置
# ==============================================================================

# 要训练的模型（逗号分隔）：NHITS, TCN, NBEATSx
MODELS="NHITS"

# 是否使用外生变量（时间特征）
USE_EXOG="--use_exog"
# 如需禁用外生变量，取消下面一行注释并注释上面一行
# USE_EXOG="--no_use_exog"

# ==============================================================================
# 6. 训练参数配置
# ==============================================================================

MAX_STEPS=5000            # 最大训练步数
LEARNING_RATE=5e-4       # 学习率
BATCH_SIZE=32             # 批次大小
WINDOWS_BATCH_SIZE=2048   # 窗口批次大小
VAL_CHECK_STEPS=200       # 验证间隔步数
EARLY_STOP_PATIENCE=8    # 早停耐心值
INTERNAL_VAL_DAYS=180      # 内部早停验证天数


# 数据缩放类型：identity（不缩放）、standard（标准化）、robust（鲁棒）、minmax（归一化）
SCALER_TYPE="robust"

# ==============================================================================
# 7. 评估参数配置
# ==============================================================================

EVAL_STEP_SIZE=1          # CV 滚动步长（默认1，每小时预测）

# 是否跳过验证集 CV（只跑测试集）
# SKIP_VAL_CV="--skip_val_cv"
SKIP_VAL_CV=""

# ==============================================================================
# 8. GPU/系统配置
# ==============================================================================

SEED=42                   # 随机种子

# GPU 配置（根据实际情况修改）

# CPU 训练（当前系统无可用 GPU）
# ACCELERATOR="cpu"
# DEVICES="auto"

# 单卡训练（推荐，有GPU时取消注释并注释上面两行）
GPU_IDS="5"
ACCELERATOR="gpu"
DEVICES="1"

# 多卡训练示例（取消注释并配置）
# GPU_IDS="0,1,2,3"
# ACCELERATOR="gpu"
# DEVICES="4"
# STRATEGY="ddp"

# 矩阵乘法精度：highest、high、medium
MATMUL_PRECISION="medium"

# ==============================================================================
# 9. 启动训练
# ==============================================================================

echo "========================================"
echo "  电价预测模型训练启动脚本"
echo "========================================"
echo "发布时间点（瑞典时间）："
echo "  - 00:00 发布 → begin=次日00:00"
echo "  - 12:00 发布 → begin=后日00:00"
echo "========================================"
echo "输出目录: $OUT_DIR"
echo "数据文件: $DATA_PATH"
echo "模型: $MODELS"
echo "发布时间: $INSURED_TIME"
echo "GPU: $GPU_IDS"
echo "========================================"

# 设置 CUDA 可见设备（如果指定了 GPU）
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "已设置 CUDA_VISIBLE_DEVICES=$GPU_IDS"
fi

# 创建输出目录
mkdir -p "$OUT_DIR"
echo "已创建输出目录: $OUT_DIR"

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
    ('$EXOG_SE2_CONSUMPTION_CSV', 'SE2_CONSUMPTION'),
    ('$EXOG_SE2_HYDRO_CSV', 'SE2_HYDRO'),
    ('$EXOG_SE2_RESIDUAL_LOAD_CSV', 'SE2_RESIDUAL_LOAD'),
    ('$EXOG_SE2_SE3_NTC_CSV', 'SE2_SE3_NTC'),
    ('$EXOG_SE2_SE3_FLOW_CSV', 'SE2_SE3_FLOW'),
    ('$EXOG_SE3_SE2_NTC_CSV', 'SE3_SE2_NTC'),
    ('$EXOG_SE3_SE2_FLOW_CSV', 'SE3_SE2_FLOW'),
]:
    if not path:
        continue
    p = os.path.join(PROJECT_DIR, path) if not path.startswith('/') else path
    if os.path.isfile(p):
        configs.append({'csv_path': p, 'name': name})
print(json.dumps(configs))
")
echo "外部外生变量: $(echo "$EXOG_CONFIGS" | python3 -c "import json,sys; cf=json.load(sys.stdin); print([c['name'] for c in cf] if cf else '无')" 2>/dev/null)"

# 启动训练
python $TRAIN_SCRIPT \
    --excel_path "$DATA_PATH" \
    --date_col "$DATE_COL" \
    --target_col "$TARGET_COL" \
    --unique_id "$UNIQUE_ID" \
    --csv_sep "$CSV_SEP" \
    --csv_encoding "$CSV_ENCODING" \
    --missing_strategy "$MISSING_STRATEGY" \
    \
    --train_start "$TRAIN_START" \
    --train_end "$TRAIN_END" \
    --val_start "$VAL_START" \
    --val_end "$VAL_END" \
    --test_start "$TEST_START" \
    --test_end "$TEST_END" \
    \
    --input_days $INPUT_DAYS \
    --issued_tz "$ISSUED_TZ" \
    --insured_time $INSURED_TIME \
    --target_hours $TARGET_HOURS \
    \
    --models "$MODELS" \
    $USE_EXOG \
    \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --windows_batch_size $WINDOWS_BATCH_SIZE \
    --val_check_steps $VAL_CHECK_STEPS \
    --early_stop_patience_steps $EARLY_STOP_PATIENCE \
    --internal_val_days $INTERNAL_VAL_DAYS \
    --nf_scaler_type "$SCALER_TYPE" \
    \
    --eval_step_size $EVAL_STEP_SIZE \
    $SKIP_VAL_CV \
    \
    --seed $SEED \
    --out_dir "$OUT_DIR" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    ${STRATEGY:+--strategy "$STRATEGY"} \
    --matmul_precision "$MATMUL_PRECISION" \
    --exog_configs "$EXOG_CONFIGS"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "  训练完成！"
    echo "  结果已保存到: $OUT_DIR"
    echo "========================================"
else
    echo "========================================"
    echo "  训练失败！"
    echo "========================================"
    exit 1
fi
