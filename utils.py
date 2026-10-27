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

def resolve_and_canonicalize_amazon_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    Resolves short Amazon links (amzn.in, amzn.to, a.co, etc.), strips tracking parameters,
    and extracts canonical Amazon product URL and 10-digit ASIN.
    """
    if not url or not isinstance(url, str):
        return False, "", None

    cleaned = url.strip()
    
    # Check if short URL needing redirect resolution
    short_domains = ("amzn.in", "amzn.to", "a.co", "amzn.eu", "amzn.asia", "amzn.com")
    if any(dom in cleaned.lower() for dom in short_domains) or ('/d/' in cleaned and '/dp/' not in cleaned):
        try:
            import requests
            r = requests.get(
                cleaned,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                allow_redirects=True,
                timeout=3.5
            )
            if r.url and r.url != cleaned:
                cleaned = r.url
        except Exception as e:
            logger.warning(f"Could not follow short link redirect: {e}")

    # Extract 10-character alphanumeric ASIN
    asin_match = re.search(r'(?:/dp/|/gp/product/|/d/|/product/|/ASIN/)([A-Z0-9]{10})', cleaned, re.IGNORECASE)
    asin = asin_match.group(1).upper() if asin_match else None

    # Identify regional domain
    domain_match = re.search(r'https?://(?:www\.)?(amazon\.[a-z\.]+|amzn\.[a-z]+|a\.co)', cleaned, re.IGNORECASE)
    domain = domain_match.group(1).lower() if domain_match else 'amazon.in'
    if 'amzn' in domain or 'a.co' in domain:
        domain = 'amazon.in'

    if asin:
        canonical = f"https://www.{domain}/dp/{asin}"
        return True, canonical, asin

    return False, cleaned, None

def validate_amazon_url(url: str, config: Dict = None) -> Tuple[bool, str]:
    """Validate Amazon product URL (supports international domains, short links, and raw ASINs)"""
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty"
    
    url = url.strip()
    is_valid, canonical, asin = resolve_and_canonicalize_amazon_url(url)
    if not is_valid or not asin:
        return False, "Please enter a valid Amazon product URL or short link (e.g. https://www.amazon.in/dp/B0HGRM4S1B or https://amzn.in/d/08obX5YK)"
    
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
    """Get theme configuration with full token coverage"""
    themes = {
        'light': {
            'bg_primary': '#f8fafc',
            'bg_secondary': '#ffffff',
            'text_primary': '#0f172a',
            'text_secondary': '#475569',
            'accent': '#4f46e5',
            'accent_hover': '#4338ca',
            'button_bg': 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
            'button_text': '#ffffff',
            'input_bg': '#ffffff',
            'input_text': '#0f172a',
            'input_border': '#cbd5e1',
            'sidebar_bg': '#f8fafc',
            'sidebar_text': '#0f172a',
            'card_bg': '#ffffff',
            'card_border': '#e2e8f0',
            'success': '#16a34a',
            'warning': '#ea580c',
            'error': '#dc2626',
            'border': '#e2e8f0'
        },
        'dark': {
            'bg_primary': '#0b0f19',
            'bg_secondary': '#1e293b',
            'text_primary': '#f8fafc',
            'text_secondary': '#94a3b8',
            'accent': '#6366f1',
            'accent_hover': '#4f46e5',
            'button_bg': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            'button_text': '#ffffff',
            'input_bg': '#1e293b',
            'input_text': '#f8fafc',
            'input_border': '#334155',
            'sidebar_bg': '#0f172a',
            'sidebar_text': '#f8fafc',
            'card_bg': '#1e293b',
            'card_border': '#334155',
            'success': '#22c55e',
            'warning': '#f97316',
            'error': '#ef4444',
            'border': '#334155'
        }
    }
    return themes.get(theme, themes['light'])
