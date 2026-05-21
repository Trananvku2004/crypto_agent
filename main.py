# main.py
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import threading
import time

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       AI Crypto Futures Paper Trading Agent                  ║
║       RL-based | PPO | Binance | Telegram | Dashboard        ║
╠══════════════════════════════════════════════════════════════╣
║  python main.py download    → Tải data lịch sử              ║
║  python main.py test-data   → Test kết nối Binance          ║
║  python main.py features    → Test feature engineering      ║
║  python main.py train       → Train PPO agent               ║
║  python main.py backtest    → Backtest trên test set        ║
║  python main.py evaluate    → So sánh Train/Val/Test        ║
║  python main.py simulate    → Mô phỏng N lệnh               ║
║  python main.py trade       → Paper trading realtime        ║
║  python main.py bot         → Khởi động Telegram bot        ║
║  python main.py dashboard   → Web dashboard                 ║
║  python main.py run         → Chạy tất cả                   ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Commands ──────────────────────────────────────────────────

def cmd_download(args):
    """Thu thập data lịch sử."""
    from data.fetch_data import download_historical
    symbols = [s.strip() for s in args.symbols.split(",")]
    if len(symbols) == 1 and symbols[0] == "ALL":
        import config
        symbols = config.SYMBOLS
    for s in symbols:
        print(f"\n{'='*50}")
        download_historical(s, days=args.days)

def cmd_test_data(args):
    """Test kết nối Binance realtime."""
    from data.fetch_data import fetch_ohlcv, fetch_current_price
    import config
    print("\nTest kết nối Binance...")
    all_ok = True
    for symbol in config.SYMBOLS:
        price = fetch_current_price(symbol)
        df    = fetch_ohlcv(symbol, limit=3)
        ok    = not df.empty and price > 0
        if not ok:
            all_ok = False
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {symbol:<12} ${price:>12,.2f}")
    print(f"\n{'✅ Tất cả kết nối OK' if all_ok else '❌ Có lỗi kết nối'}")

def cmd_features(args):
    """Test feature engineering."""
    from data.feature_engine import load_and_process, train_val_test_split
    import config
    for symbol in config.SYMBOLS:
        print(f"\n{'='*50}")
        try:
            df = load_and_process(symbol)
            train_val_test_split(df)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

def cmd_train(args):
    """Train PPO agent."""
    from agent.train import train_agent
    train_agent()

def cmd_backtest(args):
    """Backtest trên test set."""
    import backtest
    backtest.run()

def cmd_evaluate(args):
    """Đánh giá agent Train/Val/Test."""
    import evaluate
    evaluate.run()

def cmd_simulate(args):
    """Mô phỏng N lệnh trước khi realtime."""
    import simulate
    simulate.simulate(args.trades, args.symbol)

def cmd_trade(args):
    """Chạy paper trading realtime."""
    from trading.paper_engine import run_live
    run_live()

def cmd_bot(args):
    """Khởi động Telegram bot."""
    from bot.telegram_bot import start_bot
    start_bot()

def cmd_dashboard(args):
    """Khởi động web dashboard."""
    from dashboard.server import run_server
    run_server(port=args.port, debug=args.debug)

def cmd_scan(args):
    """Scan tín hiệu 30 coins."""
    from data.market_scanner import scanner
    results = scanner.scan_all()
    print(f"\n{'='*55}")
    print("  TOP SIGNALS")
    print(f"{'='*55}")
    for s in scanner.get_top_signals(5):
        print(f"  {s['symbol']:<12} {s.get('signal',''):<12} "
              f"RSI:{s.get('rsi',0):>5} "
              f"Score:{s.get('score',0):>+3}")

def cmd_run(args):
    """Chạy toàn bộ hệ thống."""
    errors = []

    # Kiểm tra import
    try:
        from trading.paper_engine import run_live
    except Exception as e:
        errors.append(f"paper_engine: {e}")

    try:
        from bot.telegram_bot import start_bot
    except Exception as e:
        errors.append(f"telegram_bot: {e}")

    try:
        from dashboard.server import run_server
    except Exception as e:
        errors.append(f"dashboard: {e}")

    if errors:
        print("[ERROR] Import thất bại:")
        for e in errors:
            print(f"  ❌ {e}")
        return

    from trading.paper_engine import run_live
    from bot.telegram_bot    import start_bot
    from dashboard.server    import run_server

    print("[SYSTEM] Khởi động toàn bộ hệ thống...")
    print("="*55)

    # Trading Engine
    t1 = threading.Thread(
        target=run_live,
        daemon=True, name="TradingEngine"
    )
    t1.start()
    print("[SYSTEM] ✅ Trading Engine started")
    time.sleep(2)

    # Telegram Bot
    t2 = threading.Thread(
        target=start_bot,
        daemon=True, name="TelegramBot"
    )
    t2.start()
    print("[SYSTEM] ✅ Telegram Bot started")
    time.sleep(1)

    # Dashboard
    t3 = threading.Thread(
        target=run_server,
        kwargs={"port": 5000},
        daemon=True, name="Dashboard"
    )
    t3.start()
    print("[SYSTEM] ✅ Dashboard started → http://localhost:5000")

    print("="*55)
    print("[SYSTEM] Tất cả services đang chạy")
    print("[SYSTEM] Nhấn Ctrl+C để dừng\n")

    try:
        while True:
            time.sleep(5)
            # Health check
            dead = []
            if not t1.is_alive(): dead.append("TradingEngine")
            if not t2.is_alive(): dead.append("TelegramBot")
            if not t3.is_alive(): dead.append("Dashboard")
            if dead:
                print(f"[WARNING] Dịch vụ đã dừng: {', '.join(dead)}")
    except KeyboardInterrupt:
        print("\n[SYSTEM] Đang dừng hệ thống...")
        print("[SYSTEM] Đã dừng. Tạm biệt!")

# ── CLI Parser ────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog        = "crypto_agent",
        description = "AI Crypto Futures Paper Trading Agent",
        formatter_class = argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # download
    p = sub.add_parser("download", help="Tải data lịch sử từ Binance")
    p.add_argument("--symbols", default="ALL",
                   help="VD: 'BTC/USDT,ETH/USDT' hoặc 'ALL'")
    p.add_argument("--days", type=int, default=90,
                   help="Số ngày lịch sử (default: 90)")
    p.set_defaults(func=cmd_download)

    # test-data
    p = sub.add_parser("test-data", help="Test kết nối Binance")
    p.set_defaults(func=cmd_test_data)

    # features
    p = sub.add_parser("features", help="Test feature engineering")
    p.set_defaults(func=cmd_features)

    # train
    p = sub.add_parser("train", help="Train PPO agent")
    p.set_defaults(func=cmd_train)

    # backtest
    p = sub.add_parser("backtest", help="Backtest trên test set")
    p.set_defaults(func=cmd_backtest)

    # evaluate
    p = sub.add_parser("evaluate", help="So sánh Train/Val/Test metrics")
    p.set_defaults(func=cmd_evaluate)

    # simulate
    p = sub.add_parser("simulate", help="Mô phỏng N lệnh trước realtime")
    p.add_argument("--trades", type=int,  default=20)
    p.add_argument("--symbol", type=str,  default="BTC/USDT")
    p.set_defaults(func=cmd_simulate)

    # trade
    p = sub.add_parser("trade", help="Paper trading realtime")
    p.set_defaults(func=cmd_trade)

    # bot
    p = sub.add_parser("bot", help="Khởi động Telegram bot")
    p.set_defaults(func=cmd_bot)

    # dashboard
    p = sub.add_parser("dashboard", help="Web dashboard Bloomberg-style")
    p.add_argument("--port",  type=int,  default=5000)
    p.add_argument("--debug", action="store_true", default=False)
    p.set_defaults(func=cmd_dashboard)

    # scan
    p = sub.add_parser("scan", help="Scan tín hiệu 30 coins")
    p.set_defaults(func=cmd_scan)

    # run
    p = sub.add_parser("run", help="Chạy toàn bộ hệ thống")
    p.set_defaults(func=cmd_run)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
    else:
        args.func(args)