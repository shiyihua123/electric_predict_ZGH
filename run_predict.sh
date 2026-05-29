#!/bin/bash
# 电价预测启动脚本
#
# 用法：
#   bash run_predict.sh                                    # 使用默认参数运行
#   bash run_predict.sh --insured_time 12 --target_hours 168  # 自定义参数
#   bash run_predict.sh --help                               # 查看帮助

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========== 默认值 ==========
DATA_PATH="sourceData/SE2_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"
DATE_COL="date"
TARGET_COL="price"
CSV_SEP=","
CSV_ENCODING="utf-8"
FREQ="h"
UNIQUE_ID="SE2"
MISSING_STRATEGY="interpolate"

# 版本名称（与 run_train.sh 保持一致）
VERSION="my_new_experiment"

MODEL_DIR="outputs/${VERSION}/models_results/neuralforecast_bundle"
ISSUED_TZ="Europe/Stockholm"
INSURED_TIME="0"
TARGET_HOURS="1056"
ISSUED_DATE="$(date +%Y-%m-%d)"
EXOG_SE3_CSV="sourceData/SE3_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"
EXOG_SE1_CSV="sourceData/SE1_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"
EXOG_SE4_CSV="sourceData/SE4_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv"
EXOG_SE2_WIND_CSV="sourceData/SE2_Wind_Power_Production_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_SOLAR_CSV="sourceData/SE2_Solar_Photovoltaic_Production_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_CONSUMPTION_CSV="sourceData/SE2_Consumption_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_RESIDUAL_LOAD_CSV="sourceData/SE2_Residual_Load_MWh_h_H_Actual/MW_latest.csv"
# EXOG_SE2_HYDRO_CSV="sourceData/SE2_Hydro_Power_Production_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_SE3_NTC_CSV="sourceData/SE2_SE3_Exchange_Net_Transfer_Capacity_MW_15min_REMIT/MW_latest_H_mean.csv"
EXOG_SE2_SE3_FLOW_CSV="sourceData/SE2_SE3_Exchange_Physical_Flow_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE3_SE2_NTC_CSV="sourceData/SE3_SE2_Exchange_Net_Transfer_Capacity_MW_15min_REMIT/MW_latest_H_mean.csv"
EXOG_SE3_SE2_FLOW_CSV="sourceData/SE3_SE2_Exchange_Physical_Flow_MWh_h_H_Actual/MW_latest.csv"

# SE2 跨境物理流
EXOG_SE2_NO4_FLOW_CSV="sourceData/SE2_NO4_Exchange_Physical_Flow_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_NO3_FLOW_CSV="sourceData/SE2_NO3_Exchange_Physical_Flow_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_SE1_FLOW_CSV="sourceData/SE2_SE1_Exchange_Physical_Flow_MWh_h_H_Actual/MW_latest.csv"

# SE2 净进口/净出口
EXOG_SE2_NET_IMPORT_CSV="sourceData/SE2_Exchange_Physical_Flow_Net_Import_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE2_NET_EXPORT_CSV="sourceData/SE2_Exchange_Physical_Flow_Net_export_MWh_h_H_Actual/MW_latest.csv"

# 瑞典全国负荷
EXOG_SE_CONSUMPTION_CSV="sourceData/SE_Consumption_MWh_h_H_Actual/MW_latest.csv"
EXOG_SE_RESIDUAL_LOAD_CSV="sourceData/SE_Residual_Load_MWh_h_H_Actual/MW_latest.csv"

# SE2 其他能源产电（2021-12-15起，数据太短暂不用）
# EXOG_SE2_OTHER_POWER_CSV="sourceData/SE2_Other_Power_Production_MWh_h_H_Actual/MW_latest.csv"
OUT_CSV=""

# 是否构造 SE3-SE2 / SE4-SE2 价差特征
USE_SPREAD="--use_spread"
# USE_SPREAD="--no-use_spread"

# ========== 解析命令行参数 ==========
print_help() {
    cat << EOF
电价预测启动脚本

用法: bash run_predict.sh [选项]

  数据参数:
    --data_path PATH           CSV 数据文件路径 (默认: sourceData/.../actual_min_to_H_true_latest.csv)
    --date_col NAME            时间列名 (默认: date)
    --target_col NAME          目标价格列名 (默认: price)
    --csv_sep SEP              CSV 分隔符 (默认: ,)
    --csv_encoding ENC         CSV 编码 (默认: utf-8)
    --freq FREQ                时间频率 (默认: h)
    --unique_id ID             时间序列唯一标识 (默认: SE2)
    --missing_strategy STRAT   缺失值处理策略: interpolate / ffill / raise (默认: interpolate)

  模型参数:
    --model_dir DIR            模型目录 (默认: outputs/models_results/neuralforecast_bundle)

  预测参数:
    --issued_tz TZ             发布时区 (默认: Europe/Stockholm)
    --issued_date DATE         发布日期 YYYY-MM-DD (不指定则自动取数据最新日期 + 1 天)
    --insured_time HOUR        发布时间点: 0 / 12 (默认: 0)
    --target_hours HOURS       目标预测小时数 (默认: 1032)

  外生变量:
    --exog_se3_csv PATH        SE3 电价数据文件路径
    --exog_se1_csv PATH        SE1 电价数据文件路径
    --exog_se4_csv PATH        SE4 电价数据文件路径
    --exog_se2_wind_csv PATH   SE2 风电数据文件路径
    --exog_se2_solar_csv PATH  SE2 太阳能数据文件路径 (不指定则不加该外生变量)

  输出参数:
    --out_csv PATH             输出 CSV 路径 (默认: outputs/predict_results/predict_issued_XX_XXXh.csv)

示例:
    bash run_predict.sh
    bash run_predict.sh --insured_time 12 --target_hours 168
    bash run_predict.sh --issued_date 2026-05-19 --insured_time 0 --target_hours 120
    bash run_predict.sh --issued_date 2026-05-19 --insured_time 0 --target_hours 120 --out_csv ./my_pred.csv
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            print_help
            ;;
        --data_path)
            DATA_PATH="$2"; shift 2 ;;
        --date_col)
            DATE_COL="$2"; shift 2 ;;
        --target_col)
            TARGET_COL="$2"; shift 2 ;;
        --csv_sep)
            CSV_SEP="$2"; shift 2 ;;
        --csv_encoding)
            CSV_ENCODING="$2"; shift 2 ;;
        --freq)
            FREQ="$2"; shift 2 ;;
        --unique_id)
            UNIQUE_ID="$2"; shift 2 ;;
        --missing_strategy)
            MISSING_STRATEGY="$2"; shift 2 ;;
        --model_dir)
            MODEL_DIR="$2"; shift 2 ;;
        --issued_tz)
            ISSUED_TZ="$2"; shift 2 ;;
        --issued_date)
            ISSUED_DATE="$2"; shift 2 ;;
        --insured_time)
            INSURED_TIME="$2"; shift 2 ;;
        --target_hours)
            TARGET_HOURS="$2"; shift 2 ;;
        --out_csv)
            OUT_CSV="$2"; shift 2 ;;
        --exog_se3_csv)
            EXOG_SE3_CSV="$2"; shift 2 ;;
        --exog_se1_csv)
            EXOG_SE1_CSV="$2"; shift 2 ;;
        --exog_se4_csv)
            EXOG_SE4_CSV="$2"; shift 2 ;;
        --exog_se2_wind_csv)
            EXOG_SE2_WIND_CSV="$2"; shift 2 ;;
        --exog_se2_solar_csv)
            EXOG_SE2_SOLAR_CSV="$2"; shift 2 ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ========== 自动推算发布日期 ==========
if [ -z "$ISSUED_DATE" ]; then
    ISSUED_DATE="$(python3 -c "
import pandas as pd
df = pd.read_csv('$DATA_PATH', sep='$CSV_SEP', encoding='$CSV_ENCODING')
last_ts = pd.to_datetime(df['$DATE_COL'].iloc[-1], utc=True)
print((last_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
")"
    echo "自动推算发布日期: $ISSUED_DATE (数据最新日期 + 1 天)"
fi
if [ -z "$OUT_CSV" ]; then
    OUT_CSV="outputs/${VERSION}/predict_results/predict_${ISSUED_DATE}_issued_$(printf '%02d' "$INSURED_TIME")_${TARGET_HOURS}h.csv"
fi

# ========== 检查 ==========
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "错误: 模型目录不存在: $MODEL_DIR"
    exit 1
fi

# ========== 构建外生变量配置 ==========
# 用 Python 动态构建外生变量配置 JSON（仅添加文件存在的变量）
EXOG_CONFIGS=$(python3 -c "
import json, os
PROJECT_DIR = '$PROJECT_DIR'
configs = []
for path, name in [
    ('${EXOG_SE3_CSV:-}', 'SE3'),
    ('${EXOG_SE1_CSV:-}', 'SE1'),
    ('${EXOG_SE4_CSV:-}', 'SE4'),
    ('${EXOG_SE2_WIND_CSV:-}', 'SE2_WIND'),
    ('${EXOG_SE2_SOLAR_CSV:-}', 'SE2_SOLAR'),
    ('${EXOG_SE2_CONSUMPTION_CSV:-}', 'SE2_CONSUMPTION'),
    ('${EXOG_SE2_HYDRO_CSV:-}', 'SE2_HYDRO'),
    ('${EXOG_SE2_RESIDUAL_LOAD_CSV:-}', 'SE2_RESIDUAL_LOAD'),
    ('${EXOG_SE2_SE3_NTC_CSV:-}', 'SE2_SE3_NTC'),
    ('${EXOG_SE2_SE3_FLOW_CSV:-}', 'SE2_SE3_FLOW'),
    ('${EXOG_SE3_SE2_NTC_CSV:-}', 'SE3_SE2_NTC'),
    ('${EXOG_SE3_SE2_FLOW_CSV:-}', 'SE3_SE2_FLOW'),
    ('${EXOG_SE2_NO4_FLOW_CSV:-}', 'SE2_NO4_FLOW'),
    ('${EXOG_SE2_NO3_FLOW_CSV:-}', 'SE2_NO3_FLOW'),
    ('${EXOG_SE2_SE1_FLOW_CSV:-}', 'SE2_SE1_FLOW'),
    ('${EXOG_SE2_NET_IMPORT_CSV:-}', 'SE2_NET_IMPORT'),
    ('${EXOG_SE2_NET_EXPORT_CSV:-}', 'SE2_NET_EXPORT'),
    ('${EXOG_SE_CONSUMPTION_CSV:-}', 'SE_CONSUMPTION'),
    ('${EXOG_SE_RESIDUAL_LOAD_CSV:-}', 'SE_RESIDUAL_LOAD'),
    ('${EXOG_SE2_OTHER_POWER_CSV:-}', 'SE2_OTHER_POWER'),
]:
    if not path:
        continue
    p = os.path.join(PROJECT_DIR, path) if not path.startswith('/') else path
    if os.path.isfile(p):
        configs.append({'csv_path': p, 'name': name})
print(json.dumps(configs))
")
echo "外部外生变量: $(echo "$EXOG_CONFIGS" | python3 -c "import json,sys; cf=json.load(sys.stdin); print([c['name'] for c in cf] if cf else '无')" 2>/dev/null)"

# ========== 运行预测 ==========
echo "=========================================="
echo "  电价预测脚本"
echo "=========================================="
echo "数据文件    : $DATA_PATH"
echo "时间列      : $DATE_COL"
echo "价格列      : $TARGET_COL"
echo "分隔符      : $CSV_SEP"
echo "编码        : $CSV_ENCODING"
echo "频率        : $FREQ"
echo "序列标识    : $UNIQUE_ID"
echo "缺失值策略  : $MISSING_STRATEGY"
echo "模型目录    : $MODEL_DIR"
echo "发布时区    : $ISSUED_TZ"
echo "发布日期    : $ISSUED_DATE"
echo "发布时间    : ${INSURED_TIME}:00 (瑞典时间)"
echo "预测小时数  : $TARGET_HOURS"
echo "输出文件    : $OUT_CSV"
echo "=========================================="

PREDICT_ARGS=(
    --data_path "$DATA_PATH"
    --date_col "$DATE_COL"
    --target_col "$TARGET_COL"
    --csv_sep "$CSV_SEP"
    --csv_encoding "$CSV_ENCODING"
    --freq "$FREQ"
    --unique_id "$UNIQUE_ID"
    --missing_strategy "$MISSING_STRATEGY"
    --model_dir "$MODEL_DIR"
    --issued_tz "$ISSUED_TZ"
    --issued_date "$ISSUED_DATE"
    --insured_time "$INSURED_TIME"
    --target_hours "$TARGET_HOURS"
    --out_csv "$OUT_CSV"
    --exog_configs "$EXOG_CONFIGS" \
    $USE_SPREAD
)

python3 predict.py "${PREDICT_ARGS[@]}"

echo ""
echo "=========================================="
echo "  预测完成！结果: $OUT_CSV"
echo "=========================================="
