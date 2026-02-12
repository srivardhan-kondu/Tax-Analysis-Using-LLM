"""
Configuration management for Tax Evasion Detection System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Kaggle Configuration
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# Application Settings
APP_TITLE = os.getenv("APP_TITLE", "Tax Evasion Detection System")
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Kolkata")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# GPT-4 Analysis Settings
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0.0"))
GPT_MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", "2000"))
GPT_CONFIDENCE_THRESHOLD = float(os.getenv("GPT_CONFIDENCE_THRESHOLD", "0.7"))

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/tax_evasion_detection.db")

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Financial Indicators Configuration
FINANCIAL_INDICATORS = [
    "sales",
    "revenue_growth",
    "profit_margin",
    "employee_growth",
    "debt_ratio",
    "operating_expenses",
    "tax_to_revenue_ratio"
]

# Risk Classification
RISK_LEVELS = {
    0: "No Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# Risk Thresholds (for rule-based validation)
RISK_THRESHOLDS = {
    "tax_to_revenue_ratio_low": 0.05,  # Suspiciously low tax ratio
    "profit_margin_high": 0.40,  # Unusually high profit margin
    "debt_ratio_high": 0.75,  # High debt ratio
    "revenue_growth_inconsistent": 0.50,  # Inconsistent with tax payments
}

def validate_config():
    """Validate required configuration"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required. Please set it in .env file")
    
    return True
