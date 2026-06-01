import argparse
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('/home/syh/workplace/PythonProject/electric_predict_ZGH')
from py_file.metric import mae, rmse, bias, wape

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='模型预测与 Montel 预测对比分析')
    parser.add_argument('--pred_csv', required=True, help='模型预测结果文件')
    parser.add_argument('--montel_csv', required=True, help='Montel预测文件')
    parser.add_argument('--out_dir', required=True, help='输出目录')
    args = parser.parse_args()

    montel_path = args.montel_csv

    # 读取数据
    montel_df = pd.read_csv(montel_path)
    model_df = pd.read_csv(args.pred_csv)

    # 提取所有 h 开头的列，并按数字部分排序
    h_cols = [col for col in montel_df.columns if col.startswith('h')]
    if not h_cols:
        raise ValueError("数据中没有找到以 'h' 开头的列！")
    h_cols_sorted = sorted(h_cols, key=lambda x: int(x[1:]))

    print(f"找到 {len(h_cols_sorted)} 个 h 列，范围从 {h_cols_sorted[0]} 到 {h_cols_sorted[-1]}")

    # 检查缺失值
    before_rows = len(montel_df)
    df_clean = montel_df.dropna(subset=h_cols_sorted)
    after_rows = len(df_clean)
    removed_rows = before_rows - after_rows
    if removed_rows > 0:
        print(f"删除了 {removed_rows} 行（这些行在 h 列中存在缺失值）。")
    else:
        print("所有行均无缺失值，无需删除。")

    # 只保留两个数据集共有的 issued_utc（交集）
    common_issued_utc = set(model_df['issued_utc']) & set(montel_df['issued_utc'])
    model_df = model_df[model_df['issued_utc'].isin(common_issued_utc)]
    montel_df = montel_df[montel_df['issued_utc'].isin(common_issued_utc)]

    # 提取数据
    Y_PRICE = model_df['y'].to_numpy().ravel()
    MODEL_PREDICTION = model_df.iloc[:, -1].to_numpy().ravel()
    MONTEL_PREDICTION = montel_df.loc[:, 'h0001':'h1056'].to_numpy().ravel()
    
    # 获取模型名称
    model_name = model_df.columns[-1]

    # 计算指标
    print("\n=== 指标计算结果 ===")
    print(f"{'type':<15} {'counts':<10} {'MAE':<12} {'RMSE':<12} {'Bias':<12} {'WAPE':<12}")
    print("-" * 75)

    # Model vs Y_PRICE
    model_counts = len(Y_PRICE)
    model_mae = mae(Y_PRICE, MODEL_PREDICTION)
    model_rmse = rmse(Y_PRICE, MODEL_PREDICTION)
    model_bias = bias(Y_PRICE, MODEL_PREDICTION)
    model_wape = wape(Y_PRICE, MODEL_PREDICTION)

    # Montel vs Y_PRICE
    montel_counts = len(MONTEL_PREDICTION)
    if len(Y_PRICE) != len(MONTEL_PREDICTION):
        min_len = min(len(Y_PRICE), len(MONTEL_PREDICTION))
        montel_mae = mae(Y_PRICE[:min_len], MONTEL_PREDICTION[:min_len])
        montel_rmse = rmse(Y_PRICE[:min_len], MONTEL_PREDICTION[:min_len])
        montel_bias = bias(Y_PRICE[:min_len], MONTEL_PREDICTION[:min_len])
        montel_wape = wape(Y_PRICE[:min_len], MONTEL_PREDICTION[:min_len])
    else:
        montel_mae = mae(Y_PRICE, MONTEL_PREDICTION)
        montel_rmse = rmse(Y_PRICE, MONTEL_PREDICTION)
        montel_bias = bias(Y_PRICE, MONTEL_PREDICTION)
        montel_wape = wape(Y_PRICE, MONTEL_PREDICTION)

    # 输出结果（先放 MONTEL）
    print(f"{'MONTEL':<15} {montel_counts:<10} {montel_mae:<12.4f} {montel_rmse:<12.4f} {montel_bias:<12.4f} {montel_wape:<12.4f}")
    print(f"{model_name:<15} {model_counts:<10} {model_mae:<12.4f} {model_rmse:<12.4f} {model_bias:<12.4f} {model_wape:<12.4f}")

    # 将结果保存为 CSV
    result_df = pd.DataFrame({
        'type': ['MONTEL', model_name],
        'counts': [montel_counts, model_counts],
        'MAE': [montel_mae, model_mae],
        'RMSE': [montel_rmse, model_rmse],
        'Bias': [montel_bias, model_bias],
        'WAPE': [montel_wape, model_wape]
    })

    # 确保输出目录存在
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    output_path = os.path.join(args.out_dir, 'metric_comparison.csv')
    result_df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    horizon = model_df['forecast_day'].max() * 24
    print(f"窗口大小: {horizon} 小时")

    # 可视化：滑动窗口展示真实值与预测值
    visualize_predictions(model_df, montel_df, Y_PRICE, MODEL_PREDICTION, MONTEL_PREDICTION, model_name, horizon, args.out_dir)


def visualize_predictions(model_df, montel_df, y_true, y_model, y_montel, model_name, horizon, out_dir):
    """
    可视化真实值与两个预测值的对比（随机挑选20个窗口）
    
    参数:
        model_df: 模型预测数据框（包含时间序列信息）
        montel_df: Montel预测数据框
        y_true: 真实值数组
        y_model: 模型预测值数组
        y_montel: Montel预测值数组
        model_name: 模型名称
        horizon: 窗口大小（小时）
        out_dir: 输出目录
    """
    # 确保输出目录存在
    viz_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 获取时间序列
    time_seq = model_df['ds'].values
    
    # 计算窗口数量
    total_length = len(y_true)
    num_windows = total_length // horizon
    num_samples = min(20, num_windows)  # 最多取20个
    
    if num_windows == 0:
        print("数据长度不足一个窗口，跳过可视化")
        return
    
    # 随机挑选20个窗口
    np.random.seed(42)  # 设置随机种子，结果可重复
    selected_windows = np.random.choice(num_windows, size=num_samples, replace=False)
    selected_windows = sorted(selected_windows)  # 按顺序排列
    
    print(f"\n=== 开始可视化 ===")
    print(f"总数据点: {total_length}")
    print(f"窗口大小: {horizon} 小时")
    print(f"总窗口数: {num_windows}")
    print(f"随机挑选: {num_samples} 个窗口")
    
    # 为选中的窗口生成可视化
    for idx, i in enumerate(selected_windows):
        # 计算窗口范围
        start_idx = i * horizon
        end_idx = start_idx + horizon
        
        # 提取窗口数据
        window_time = time_seq[start_idx:end_idx]
        window_true = y_true[start_idx:end_idx]
        window_model = y_model[start_idx:end_idx]
        window_montel = y_montel[start_idx:end_idx]
        
        # 创建画布
        plt.figure(figsize=(16, 8))
        
        # 绘制三条曲线
        plt.plot(window_time, window_true, label='Actual (y)', color='blue', linewidth=2)
        plt.plot(window_time, window_model, label=f'{model_name} Prediction', color='green', linewidth=2)
        plt.plot(window_time, window_montel, label='Montel Prediction', color='orange', linewidth=2)
        
        # 设置图表属性
        plt.title(f'Price Prediction Comparison - Window {i+1}', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (EUR/MWh)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置横坐标只显示几个关键点（约每7天显示一个）
        num_ticks = min(7, len(window_time))  # 最多显示7个时间点
        tick_indices = np.linspace(0, len(window_time)-1, num_ticks, dtype=int)
        plt.xticks(tick_indices, window_time[tick_indices], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(viz_dir, f'window_{i+1:03d}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 打印进度
        print(f"已生成 {idx+1}/{num_samples} 个窗口图像 (窗口 {i+1})")
    
    print(f"\n可视化完成！图像已保存到: {viz_dir}")


if __name__ == '__main__':
    main()
