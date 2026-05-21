# evaluate.py
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from stable_baselines3   import PPO
from env.trading_env     import CryptoTradingEnv
from data.feature_engine import load_and_process, train_val_test_split
import config

def run():
    print("\n" + "="*60)
    print("  PHASE: EVALUATE — Train / Val / Test")
    print("="*60)

    # Load data
    df = load_and_process("BTC/USDT")
    train_df, val_df, test_df = train_val_test_split(df)

    # Load model
    model_path = "models/best_model.zip"
    if not os.path.exists(model_path):
        print("[ERROR] Chưa có model. Chạy: python main.py train")
        return
    model = PPO.load(model_path)

    results = {}
    for name, split_df in [("Train", train_df),
                            ("Val",   val_df),
                            ("Test",  test_df)]:
        env  = CryptoTradingEnv(split_df, mode="test")
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        # Force close nếu còn vị thế
        if env.position != 0:
            price    = float(split_df["close"].iloc[-1])
            raw      = (price - env.entry_price) / env.entry_price * env.position
            pnl      = raw * env.position_size - config.TRADE_FEE * env.position_size * 2
            env.trades.append({"pnl": pnl, "entry": env.entry_price,
                                "exit": price, "side": env.position})
        results[name] = env.get_metrics()

    # In bảng so sánh
    print("\n" + "─"*65)
    print(f"  {'Metric':<18} {'Train':>12} {'Val':>12} {'Test':>12}")
    print("─"*65)
    for metric in ["winrate","total_pnl","n_trades",
                   "sharpe","max_drawdown","profit_factor"]:
        row = f"  {metric:<18}"
        for name in ["Train","Val","Test"]:
            v = results[name].get(metric, 0)
            if metric == "total_pnl":
                row += f" {v:>+11.4f} "
            elif metric in ["winrate","max_drawdown"]:
                row += f" {v:>10.2f}%"
            else:
                row += f" {v:>12}"
        print(row)
    print("─"*65)

    # Kiểm tra overfit
    train_wr = results["Train"]["winrate"]
    test_wr  = results["Test"]["winrate"]
    train_sh = results["Train"]["sharpe"]
    test_sh  = results["Test"]["sharpe"]

    print("\n📊 Nhận xét:")
    if test_wr >= 50:
        print(f"  ✅ Winrate Test {test_wr}% — đạt mục tiêu")
    else:
        print(f"  ⚠️  Winrate Test {test_wr}% — cần cải thiện")

    if test_sh > 1.0:
        print(f"  ✅ Sharpe Test {test_sh} — xuất sắc")
    elif test_sh > 0:
        print(f"  ⚠️  Sharpe Test {test_sh} — chấp nhận được")
    else:
        print(f"  ❌ Sharpe Test {test_sh} — cần train thêm")

    if train_wr > test_wr * 1.3:
        print(f"  ⚠️  Có dấu hiệu OVERFIT (Train {train_wr}% vs Test {test_wr}%)")
    else:
        print(f"  ✅ Không overfit — model generalize tốt")

if __name__ == "__main__":
    run()