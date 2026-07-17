"""
Utility functions for Amazon Review Insight Pro
"""
import re
import hashlib
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import yaml

# Configure logging
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()

def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'scraping': {
            'max_reviews_per_page': 50,
            'retry_attempts': 3,
            'timeout_seconds': 30,
            'cache_duration_hours': 24,
            'supported_domains': ['amazon.in', 'amazon.com', 'amazon.co.uk', 'amazon.ca']
        },
        'nlp': {
            'sentiment_threshold_positive': 0.2,
            'sentiment_threshold_negative': -0.2,
            'authenticity_threshold_high': 70,
            'authenticity_threshold_medium': 40
        }
    }

def validate_amazon_url(url: str, config: Dict) -> Tuple[bool, str]:
    """Validate if the URL is a valid Amazon product URL"""
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty"
    
    # Basic URL format validation
    supported_domains = '|'.join(config['scraping']['supported_domains'])
    url_pattern = rf'^https?://(www\.)?({supported_domains})/.*'
    if not re.match(url_pattern, url.strip()):
        return False, f"Please enter a valid Amazon URL from supported domains: {', '.join(config['scraping']['supported_domains'])}"
    
    # Check if it's a product page
    if '/dp/' not in url and '/product/' not in url:
        return False, "Please enter a direct product page URL (containing /dp/ or /product/)"
    
    return True, "Valid URL"

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS and other attacks"""
    if not text:
        return ""
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    return text.strip()

def get_url_hash(url: str) -> str:
    """Generate a hash for URL caching"""
    return hashlib.md5(url.encode()).hexdigest()

@contextmanager
def get_db_connection():
    """Get database connection for caching"""
    conn = sqlite3.connect('review_cache.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            url_hash TEXT PRIMARY KEY,
            url TEXT,
            data TEXT,
            created_at TIMESTAMP,
            expires_at TIMESTAMP
        )
    ''')
    yield conn
    conn.close()

def get_cached_data(url: str) -> Optional[Dict]:
    """Retrieve cached data for a URL"""
    try:
        with get_db_connection() as conn:
            url_hash = get_url_hash(url)
            cursor = conn.execute(
                'SELECT data FROM cache WHERE url_hash = ? AND expires_at > ?',
                (url_hash, datetime.now())
            )
            result = cursor.fetchone()
            if result:
                logger.info(f"Cache hit for URL: {url}")
                return json.loads(result[0])
    except Exception as e:
        logger.error(f"Error retrieving cached data: {e}")
    return None

def cache_data(url: str, data: Dict, config: Dict) -> None:
    """Cache data for a URL"""
    try:
        with get_db_connection() as conn:
            url_hash = get_url_hash(url)
            expires_at = datetime.now() + timedelta(hours=config['scraping']['cache_duration_hours'])
            conn.execute(
                'INSERT OR REPLACE INTO cache (url_hash, url, data, created_at, expires_at) VALUES (?, ?, ?, ?, ?)',
                (url_hash, url, json.dumps(data), datetime.now(), expires_at)
            )
            conn.commit()
            logger.info(f"Data cached for URL: {url}")
    except Exception as e:
        logger.error(f"Error caching data: {e}")

def format_number(num: float, decimals: int = 1) -> str:
    """Format number with specified decimal places"""
    return f"{num:.{decimals}f}"

def get_theme_config(theme: str = 'light') -> Dict:
    """Get theme configuration"""
    themes = {
        'light': {
            'bg_primary': '#f9fafc',
            'bg_secondary': '#ffffff',
            'text_primary': '#2a3f5f',
            'text_secondary': '#666666',
            'accent': '#6e8efb',
            'success': '#00AA45',
            'warning': '#FFA500',
            'error': '#FF4D4D',
            'border': '#e1e5e9'
        },
        'dark': {
            'bg_primary': '#0e1117',
            'bg_secondary': '#262730',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'accent': '#6e8efb',
            'success': '#00AA45',
            'warning': '#FFA500',
            'error': '#FF4D4D',
            'border': '#3a3a3a'
        }
    }
    return themes.get(theme, themes['light'])
