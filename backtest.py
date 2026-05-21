# backtest.py
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from stable_baselines3        import PPO
from data.feature_engine      import load_and_process, train_val_test_split
from env.trading_env          import CryptoTradingEnv
import config

def run():
    print("\n" + "="*60)
    print("  PHASE: BACKTEST ON TEST SET")
    print("="*60)

    # ── Load data ─────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df = load_and_process("BTC/USDT")
    _, _, test_df = train_val_test_split(df)
    print(f"  Test rows: {len(test_df)}")

    # ── Load model ────────────────────────────────────────
    model_path = "models/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "models/ppo_crypto_agent_final.zip"
    if not os.path.exists(model_path):
        print("[ERROR] Chưa có model.")
        return

    print(f"\n[2/4] Loading model: {model_path}")
    model = PPO.load(model_path)

    # ── Chạy backtest thủ công ────────────────────────────
    print("\n[3/4] Running backtest...")
    env = CryptoTradingEnv(test_df, mode="test")
    obs, _ = env.reset()

    balance_history = [config.INITIAL_BALANCE]
    action_counts   = {0:0, 1:0, 2:0, 3:0}
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action    = int(action)
        action_counts[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        balance_history.append(info["balance"] +
            (env.position_size if env.position != 0 else 0))
        done = terminated or truncated
        step += 1

    # Đóng vị thế cuối nếu còn
    if env.position != 0:
        last_price = test_df["close"].iloc[min(env.current_step, len(test_df)-1)]
        raw_pnl    = (last_price - env.entry_price) / env.entry_price * env.position
        pnl_usdt   = raw_pnl * env.position_size - config.TRADE_FEE * env.position_size * 2
        env.balance += env.position_size + pnl_usdt
        env.trades.append({"pnl": pnl_usdt, "entry": env.entry_price,
                            "exit": last_price, "side": env.position})
        env.position = 0

    # ── Metrics ───────────────────────────────────────────
    print("\n[4/4] Computing metrics...")
    metrics = env.get_metrics()

    final_balance = env.balance
    total_return  = (final_balance - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100

    print("\n" + "─"*45)
    print("  BACKTEST RESULTS (Test Set — Unseen Data)")
    print("─"*45)
    print(f"  Steps        : {step}")
    print(f"  Trades       : {metrics['n_trades']}")
    print(f"  Winrate      : {metrics['winrate']}%")
    print(f"  Total PnL    : {metrics['total_pnl']:+.4f} USDT")
    print(f"  Sharpe Ratio : {metrics['sharpe']}")
    print(f"  Max Drawdown : {metrics['max_drawdown']}%")
    print(f"  Profit Factor: {metrics['profit_factor']}")
    print(f"  Total Return : {total_return:+.2f}%")
    print(f"  Final Balance: {final_balance:.2f} USDT")
    print("─"*45)
    print(f"\n  Action distribution:")
    names = {0:"HOLD", 1:"LONG", 2:"SHORT", 3:"CLOSE"}
    for a, cnt in action_counts.items():
        pct = cnt/step*100
        bar = "█" * int(pct/2)
        print(f"  {names[a]:<5}: {cnt:>5} ({pct:>5.1f}%) {bar}")
    print("─"*45)

    # ── Equity curve ──────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Plot 1: Equity curve
    axes[0].plot(balance_history, color="royalblue", linewidth=1.5, label="Balance")
    axes[0].axhline(y=config.INITIAL_BALANCE, color="gray",
                    linestyle="--", linewidth=1, label="Initial")
    axes[0].fill_between(range(len(balance_history)),
                          config.INITIAL_BALANCE, balance_history,
                          where=[b >= config.INITIAL_BALANCE for b in balance_history],
                          alpha=0.2, color="green")
    axes[0].fill_between(range(len(balance_history)),
                          config.INITIAL_BALANCE, balance_history,
                          where=[b < config.INITIAL_BALANCE for b in balance_history],
                          alpha=0.2, color="red")
    axes[0].set_title("Equity Curve — Backtest (Test Set)")
    axes[0].set_ylabel("Balance (USDT)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Trade PnL bar chart
    if env.trades:
        pnls   = [t["pnl"] for t in env.trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        axes[1].bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.8)
        axes[1].axhline(y=0, color="black", linewidth=0.8)
        axes[1].set_title(f"Individual Trade PnL ({len(pnls)} trades)")
        axes[1].set_xlabel("Trade #")
        axes[1].set_ylabel("PnL (USDT)")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/backtest_result.png", dpi=150)
    plt.show()
    print("\n[SAVED] logs/backtest_result.png")

if __name__ == "__main__":
    run()