#!/usr/bin/env python3
"""列出 NeuralForecast 中所有可用的模型"""
import importlib
import inspect

import neuralforecast.models as nf_models

print("=" * 60)
print("NeuralForecast 可用模型")
print("=" * 60)

for name in sorted(dir(nf_models)):
    if name.startswith("_"):
        continue
    obj = getattr(nf_models, name)
    if not inspect.isclass(obj):
        continue
    mod = obj.__module__
    if "neuralforecast" not in mod:
        continue

    supports_exog = None
    try:
        sig = inspect.signature(obj.__init__)
        params = sig.parameters
        has_futr = "futr_exog_list" in params
        has_hist = "hist_exog_list" in params
        if has_futr and has_hist:
            supports_exog = "futr + hist"
        elif has_futr:
            supports_exog = "futr only"
        elif has_hist:
            supports_exog = "hist only"
        else:
            supports_exog = "无"
    except Exception:
        supports_exog = "?"

    print(f"  {name:25s}  外生变量: {supports_exog}")

print("=" * 60)
print(f"总数: {sum(1 for n in dir(nf_models) if not n.startswith('_') and inspect.isclass(getattr(nf_models, n)) and 'neuralforecast' in getattr(nf_models, n).__module__)}")
