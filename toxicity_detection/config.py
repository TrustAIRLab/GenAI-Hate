"""
Configuration for Perspective API

This file contains configuration settings for the Perspective API integration.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
PERSPECTIVE_API_KEY = os.getenv('PERSPECTIVE_API_KEY')

# Rate Limiting
# Perspective API upgraded tier: 10 QPS (query per second)
REQUESTS_PER_SECOND = 15.0
DELAY_BETWEEN_REQUESTS = 1.0 / REQUESTS_PER_SECOND  # 0.125 seconds

# Batch Processing
BATCH_SIZE = 100  # Number of comments to process in one batch
MAX_RETRIES = 3   # Maximum number of retries for failed requests
RETRY_DELAY = 5   # Delay in seconds before retrying

# Attributes to Analyze
# Full list of available attributes
DEFAULT_ATTRIBUTES = [
    'TOXICITY',              # Overall toxicity
    'SEVERE_TOXICITY',       # Severe toxicity
    'IDENTITY_ATTACK',       # Attacks on identity (race, religion, etc.)
    'INSULT',                # Insulting language
    'PROFANITY',             # Swear words
    'THREAT',                # Threats of violence
]

# Optional attributes (may have lower coverage)
OPTIONAL_ATTRIBUTES = [
    'SEXUALLY_EXPLICIT',     # Sexual content
    'FLIRTATION',            # Flirtatious language
]


# Language Configuration
DEFAULT_LANGUAGES = ['en']  # English by default

# Database Configuration (for future integration)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'your_database'),
    'user': os.getenv('DB_USER', 'your_user'),
    'password': os.getenv('DB_PASSWORD', ''),
}

# Table and Column Configuration
COMMENT_TABLE = 'comments'        # Table containing comments
COMMENT_COLUMN = 'comment'        # Column with comment text
ID_COLUMN = 'id'                  # Primary key column

# Output Configuration
OUTPUT_DIR = 'perspective_results'
LOG_DIR = 'logs'
