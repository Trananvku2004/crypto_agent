# simulate.py
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3   import PPO
from env.trading_env     import CryptoTradingEnv
from data.feature_engine import load_and_process, train_val_test_split
import config

def simulate(n_trades_target: int = 20, symbol: str = "BTC/USDT"):
    print("\n" + "="*60)
    print(f"  SIMULATE — Target: {n_trades_target} trades | {symbol}")
    print("="*60)

    df = load_and_process(symbol)
    _, _, test_df = train_val_test_split(df)

    model_path = "models/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "models/ppo_crypto_agent_final.zip"
    if not os.path.exists(model_path):
        print("[ERROR] Chưa có model. Chạy: python main.py train")
        return

    model = PPO.load(model_path)
    env   = CryptoTradingEnv(test_df, mode="test")
    obs, _ = env.reset()

    print(f"\n{'─'*60}")
    print(f"  {'#':<4} {'Side':<6} {'Entry':>10} "
          f"{'Exit':>10} {'PnL':>10} {'Balance':>10}")
    print(f"{'─'*60}")

    done        = False
    prev_trades = 0

    while not done and len(env.trades) < n_trades_target:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        if len(env.trades) > prev_trades:
            t    = env.trades[-1]
            side = "LONG" if t["side"] == 1 else "SHORT"
            bal  = config.INITIAL_BALANCE + sum(x["pnl"] for x in env.trades)
            icon = "✅" if t["pnl"] > 0 else "❌"
            print(f"  {icon} {len(env.trades):<3} {side:<6} "
                  f"{t['entry']:>10.2f} {t['exit']:>10.2f} "
                  f"{t['pnl']:>+10.4f} {bal:>10.2f}")
            prev_trades = len(env.trades)

    # Force close nếu còn vị thế
    if env.position != 0:
        price = float(test_df["close"].iloc[min(env.current_step, len(test_df)-1)])
        env._do_close(price)
        if env.trades:
            t   = env.trades[-1]
            bal = config.INITIAL_BALANCE + sum(x["pnl"] for x in env.trades)
            print(f"  🔄 {len(env.trades):<3} CLOSE "
                  f"{t['entry']:>10.2f} {t['exit']:>10.2f} "
                  f"{t['pnl']:>+10.4f} {bal:>10.2f}")

    # Metrics
    metrics = env.get_metrics()
    pnls    = [t["pnl"] for t in env.trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]

    print(f"\n{'='*60}")
    print(f"  RESULTS ({len(env.trades)} trades)")
    print(f"{'='*60}")
    print(f"  Initial Balance : {config.INITIAL_BALANCE:.2f} USDT")
    print(f"  Final Balance   : {config.INITIAL_BALANCE + sum(pnls):.4f} USDT")
    print(f"  Total PnL       : {sum(pnls):+.4f} USDT")
    print(f"  Total Return    : {sum(pnls)/config.INITIAL_BALANCE*100:+.4f}%")
    print(f"{'─'*60}")
    print(f"  Winrate         : {metrics['winrate']}%")
    print(f"  Avg Win         : {np.mean(wins):+.4f}" if wins else "  Avg Win  : N/A")
    print(f"  Avg Loss        : {np.mean(losses):+.4f}" if losses else "  Avg Loss : N/A")
    print(f"  Sharpe Ratio    : {metrics['sharpe']}")
    print(f"  Max Drawdown    : {metrics['max_drawdown']}%")
    print(f"  Profit Factor   : {metrics['profit_factor']}")
    print(f"{'─'*60}")

    # Checklist
    print(f"\n  📋 Checklist trước khi realtime:")
    checks = [
        (metrics['winrate']      >= 50,  f"Winrate {metrics['winrate']}% >= 50%"),
        (metrics['sharpe']       >= 0.5, f"Sharpe {metrics['sharpe']} >= 0.5"),
        (metrics['max_drawdown'] <= 10,  f"Max DD {metrics['max_drawdown']}% <= 10%"),
        (metrics['profit_factor']>= 1.0, f"Profit Factor {metrics['profit_factor']} >= 1.0"),
        (sum(pnls)               >  0,   f"Total PnL {sum(pnls):+.4f} > 0"),
    ]
    passed = sum(1 for ok, _ in checks if ok)
    for ok, msg in checks:
        print(f"  {'✅' if ok else '❌'} {msg}")

    print(f"\n  Kết quả: {passed}/{len(checks)} tiêu chí đạt")
    if passed >= 4:
        print("  🟢 SẴN SÀNG realtime!")
    elif passed >= 3:
        print("  🟡 Cân nhắc — có thể realtime thử")
    else:
        print("  🔴 Chưa sẵn sàng — train thêm")
    print("="*60)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=int, default=20)
    p.add_argument("--symbol", type=str, default="BTC/USDT")
    args = p.parse_args()
    simulate(args.trades, args.symbol)