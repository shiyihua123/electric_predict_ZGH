# -*- coding: utf-8 -*-
"""
电价预测模型训练入口

该脚本是训练的入口点，直接调用 train.py 中的主函数执行完整训练流程。

使用方法：
    # 默认配置运行（CPU）
    python main.py
    
    # 使用 GPU
    python main.py --gpu_ids=0 --accelerator=gpu --devices=1
    
    # 指定模型
    python main.py --models=NHITS,TCN
    
    # 查看所有参数
    python main.py --help
"""

from train import main, build_parser


if __name__ == "__main__":
    # 解析命令行参数并执行训练
    main(build_parser().parse_args())
