import os
from dotenv import load_dotenv

# Đọc .env từ thư mục gốc project — tuyệt đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Binance
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET        = True

# Coins
SYMBOLS   = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
TIMEFRAME = "15m"
LOOKBACK  = 200

# Vốn ảo
INITIAL_BALANCE = 1000.0
TRADE_FEE       = 0.0004

# Risk
MAX_RISK_PER_TRADE = 0.005
DAILY_MAX_LOSS     = 0.03
MAX_POSITIONS      = 2
SL_PERCENT         = 0.01
TP_PERCENT         = 0.02

# RL
TRAIN_TIMESTEPS = 3_000_000
WINDOW_SIZE     = 40
N_ENVS          = 4

# Telegram
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Paths
MODEL_PATH = "models/ppo_crypto_agent.zip"
DB_PATH    = "db/trades.db"
LOG_PATH   = "logs/agent.log"