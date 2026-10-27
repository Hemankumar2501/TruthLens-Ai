import streamlit as st
import pandas as pd
import plotly.express as px
import time
import base64
import logging
from datetime import datetime
import numpy as np
import re

# Import our modular components
from utils import load_config, validate_amazon_url, resolve_and_canonicalize_amazon_url, sanitize_input, get_cached_data, cache_data, get_theme_config
from scraper import AmazonScraper, extract_reviews, parse_review
from nlp_analyzer import NLPAnalyzer
from data_pipeline import DataCleaningPipeline
from sentiment_evaluator import SentimentEvaluator
from product_comparator import ProductComparator


# Load configuration
config = load_config()

# ========== MUST BE FIRST COMMAND ==========
st.set_page_config(
    page_title="🔍 TruthLens AI — Amazon Review & Fake Detection Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Configuration and Setup ==========
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
# Initialize cached components (eliminates 4.8s reload delay on every interaction)
@st.cache_resource(show_spinner=False)
def get_cached_nlp_analyzer():
    return NLPAnalyzer(config)

@st.cache_resource(show_spinner=False)
def get_cached_scraper():
    return AmazonScraper(config)

scraper = get_cached_scraper()
nlp_analyzer = get_cached_nlp_analyzer()


# ========== Enhanced Review Scraping Functions ==========

# ========== Enhanced NLP Analysis Functions ==========

# ========== Theme and UI Configuration ==========

# ========== Enhanced CSS with Dark Mode Support ==========
def inject_custom_css():
    theme = get_theme_config(st.session_state.get('theme', 'light'))
    st.markdown(f"""
    <style>
        /* Main styling with theme support */
        .stApp {{
            background-color: {theme['bg_primary']};
            color: {theme['text_primary']};
        }}
        
        /* Dark mode toggle button */
        .theme-toggle {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: {theme['bg_secondary']};
            border: 1px solid {theme['border']};
            border-radius: 50px;
            padding: 8px 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .theme-toggle:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .metric-card {{
                margin-bottom: 15px;
            }}
            
            .review-card {{
                padding: 15px;
                margin-bottom: 15px;
            }}
            
            .theme-toggle {{
                top: 10px;
                right: 10px;
                padding: 6px 12px;
                font-size: 0.8rem;
            }}
        }}
        
        /* Header animation */
        .header-animation {{
            animation: fadeIn 1s ease-in-out;
        }}
        
        /* Metric cards with theme support */
        .metric-card {{
            background: {theme['bg_secondary']};
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            animation: slideUp 0.5s ease-out;
            border: 1px solid {theme['border']};
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            margin: 8px 0;
            line-height: 1.2;
        }}
        
        /* Review cards with theme support */
        .review-card {{
            background: {theme['bg_secondary']};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s ease-out;
            border: 1px solid {theme['border']};
        }}
        
        .review-genuine {{
            border-left: 4px solid #00AA45;
        }}
        
        .review-fake {{
            border-left: 4px solid #FF4D4D;
        }}
        
        .review-tag {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 10px;
        }}
        
        .tag-genuine {{
            background-color: #E6F7EE;
            color: #00AA45;
        }}
        
        .tag-fake {{
            background-color: #FFEEEE;
            color: #FF4D4D;
        }}
        
        /* Timeline in sidebar */
        .timeline {{
            position: relative;
            padding-left: 20px;
        }}
        
        .timeline:before {{
            content: '';
            position: absolute;
            left: 6px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #6E8EFB;
        }}
        
        .timeline-item {{
            position: relative;
            margin-bottom: 20px;
        }}
        
        .timeline-item:before {{
            content: '';
            position: absolute;
            left: -20px;
            top: 5px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #6E8EFB;
        }}
        
        /* Accordion in sidebar */
        .accordion {{
            margin: 10px 0;
        }}
        
        .accordion-item {{
            margin-bottom: 10px;
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        
        .accordion-header {{
            padding: 10px 14px;
            background: {theme['bg_primary']} !important;
            color: {theme['text_primary']} !important;
            font-weight: 600;
            font-size: 0.9rem;
            border-bottom: 1px solid {theme['border']};
        }}
        
        .accordion-content {{
            padding: 10px 14px;
            max-height: none !important;
            color: {theme['text_secondary']} !important;
            font-size: 0.82rem;
            line-height: 1.45;
            background: {theme['card_bg']} !important;
        }}

        /* Universal Button Styling (fixes black button & invisible text issue) */
        .stButton > button,
        button[kind="primary"] {{
            background: {theme['button_bg']} !important;
            color: {theme['button_text']} !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.25rem !important;
            font-size: 0.95rem !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
            transition: all 0.2s ease !important;
        }}
        
        .stButton > button:hover,
        button[kind="primary"]:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
            opacity: 0.95 !important;
            color: #ffffff !important;
        }}

        /* Secondary Button (Theme Toggle & Sub-actions) */
        button[kind="secondary"],
        .stButton > button[kind="secondary"] {{
            background: {theme['bg_secondary']} !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        }}

        button[kind="secondary"]:hover,
        .stButton > button[kind="secondary"]:hover {{
            background: {theme['border']} !important;
            color: {theme['text_primary']} !important;
            border-color: {theme['accent']} !important;
        }}

        /* Text Input & BaseWeb Input Styling */
        div[data-testid="stTextInput"] input {{
            background-color: {theme['input_bg']} !important;
            color: {theme['input_text']} !important;
            border: 1px solid {theme['input_border']} !important;
            border-radius: 8px !important;
            font-size: 0.95rem !important;
            padding: 10px 14px !important;
        }}

        div[data-testid="stTextInput"] input::placeholder {{
            color: {theme['text_secondary']} !important;
            opacity: 0.7 !important;
        }}

        div[data-testid="stTextInput"] div[data-baseweb="input"],
        div[data-testid="stTextInput"] div[data-baseweb="base-input"] {{
            background-color: {theme['input_bg']} !important;
            border-color: {theme['input_border']} !important;
            border-radius: 8px !important;
        }}

        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] label p {{
            color: {theme['text_primary']} !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {theme['sidebar_bg']} !important;
            border-right: 1px solid {theme['border']} !important;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] strong {{
            color: {theme['sidebar_text']} !important;
        }}

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] li {{
            color: {theme['text_secondary']} !important;
        }}

        /* Streamlit Tabs */
        div[data-testid="stTabs"] button[role="tab"] {{
            color: {theme['text_secondary']} !important;
            background: transparent !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            border-bottom: 2px solid transparent !important;
            padding: 10px 18px !important;
            transition: all 0.2s ease !important;
        }}

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: {theme['accent']} !important;
            border-bottom: 2px solid {theme['accent']} !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stTabs"] button[role="tab"]:hover {{
            color: {theme['accent']} !important;
        }}

        /* Download Buttons */
        div[data-testid="stDownloadButton"] > button {{
            background: {theme['bg_secondary']} !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }}

        div[data-testid="stDownloadButton"] > button:hover {{
            background: {theme['accent']} !important;
            color: #ffffff !important;
            border-color: {theme['accent']} !important;
        }}

        /* Metrics */
        div[data-testid="stMetricValue"] {{
            color: {theme['text_primary']} !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stMetricLabel"] {{
            color: {theme['text_secondary']} !important;
        }}
        
        /* Lightweight static accents (no continuous GPU repaint lag) */
        .blob {{
            position: absolute;
            border-radius: 50%;
            opacity: 0.15;
            z-index: -1;
            pointer-events: none;
        }}
        
        .floating {{
            opacity: 0.12;
        }}

        /* Product Showcase Card with Live Scraped Image */
        .product-showcase-card {{
            background: {theme['bg_secondary']};
            border: 1px solid {theme['border']};
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.06);
            display: flex;
            gap: 24px;
            align-items: center;
            transition: all 0.3s ease;
            animation: fadeIn 0.6s ease-out;
        }}
        
        .product-showcase-card:hover {{
            box-shadow: 0 10px 28px rgba(110, 142, 251, 0.15);
            border-color: rgba(110, 142, 251, 0.4);
        }}

        .product-image-container {{
            flex-shrink: 0;
            width: 160px;
            height: 160px;
            background: #ffffff;
            border-radius: 14px;
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.07);
            border: 1px solid {theme['border']};
            overflow: hidden;
        }}

        .product-image-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.3s ease;
        }}

        .product-image-container:hover img {{
            transform: scale(1.08);
        }}

        .product-details {{
            flex-grow: 1;
        }}

        .product-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 8px;
        }}

        .badge-verified-store {{
            background: rgba(110, 142, 251, 0.15);
            color: #4a6cf7;
            border: 1px solid rgba(110, 142, 251, 0.3);
        }}

        .badge-price {{
            background: rgba(0, 170, 69, 0.12);
            color: #00aa45;
            font-size: 0.95rem;
            font-weight: 800;
            border: 1px solid rgba(0, 170, 69, 0.25);
        }}

        @media (max-width: 768px) {{
            .product-showcase-card {{
                flex-direction: column;
                text-align: center;
                padding: 18px;
            }}
            .product-image-container {{
                width: 140px;
                height: 140px;
                margin: 0 auto 12px auto;
            }}
        }}
        
        /* Keyframes */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
    </style>
    """, unsafe_allow_html=True)

# Inject custom CSS
inject_custom_css()

# ========== Theme Toggle Component ==========
def render_theme_toggle():
    """Render theme toggle button with clear label and secondary type"""
    theme = st.session_state.get('theme', 'light')
    theme_icon = "🌙" if theme == 'light' else "☀️"
    next_theme = "dark" if theme == 'light' else "light"
    label_text = "Dark Mode" if theme == 'light' else "Light Mode"
    
    if st.button(f"{theme_icon} Switch to {label_text}", key="theme_toggle", type="secondary"):
        st.session_state.theme = next_theme
        st.rerun()

# ========== Enhanced App Header with Theme Support ==========
def render_header():
    theme = get_theme_config(st.session_state.get('theme', 'light'))
    st.markdown(f"""
    <div class="header-animation">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 15px;">
                <path d="M20 40C31.0457 40 40 31.0457 40 20C40 8.9543 31.0457 0 20 0C8.9543 0 0 8.9543 0 20C0 31.0457 8.9543 40 20 40Z" fill="{theme['accent']}"/>
                <path d="M26.6667 13.3333H13.3333C11.4924 13.3333 10 14.8257 10 16.6667V23.3333C10 25.1743 11.4924 26.6667 13.3333 26.6667H26.6667C28.5076 26.6667 30 25.1743 30 23.3333V16.6667C30 14.8257 28.5076 13.3333 26.6667 13.3333Z" fill="white"/>
                <path d="M20 23.3333C21.8409 23.3333 23.3333 21.8409 23.3333 20C23.3333 18.159 21.8409 16.6667 20 16.6667C18.159 16.6667 16.6667 18.159 16.6667 20C16.6667 21.8409 18.159 23.3333 20 23.3333Z" fill="{theme['accent']}"/>
            </svg>
            <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, {theme['accent']} 0%, #a777e3 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TruthLens AI</h1>
        </div>
        <p style="color: {theme['text_secondary']}; font-size: 1.1rem; margin-bottom: 2rem; max-width: 800px;">
            <strong>Instant Amazon Review & Fake Detection Intelligence.</strong> Unmask authentic customer opinions, evaluate verified sentiment, and detect deceptive reviews with AI-powered analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

render_header()

# ========== Product Showcase Component ==========
def render_product_showcase(product_info: dict, df: pd.DataFrame):
    """Renders a sleek, modern Product Showcase Card with the scraped product image, price, and specs"""
    target_url = (product_info.get('url') if product_info else '') or st.session_state.get('last_analyzed_url', '') or st.session_state.get('url_input', '')
    asin_m = re.search(r'/(?:dp|gp/product|product)/([A-Z0-9]{10})', target_url or '', re.I)
    asin = asin_m.group(1).upper() if asin_m else ""
    
    prod_title = (product_info.get('title') if product_info else '') or (df['Product'].iloc[0] if 'Product' in df.columns and not df.empty else "")
    if (not prod_title or prod_title.lower().startswith("amazon product")) and target_url:
        slug_m = re.search(r'amazon\.[a-z\.]+/([^/]+)/(?:dp|gp/product)', target_url)
        if slug_m and not any(x in slug_m.group(1).lower() for x in ['dp', 'gp', 'product']):
            prod_title = slug_m.group(1).replace('-', ' ').strip().title()
        elif asin:
            prod_title = f"Amazon Product ({asin})"
    if not prod_title:
        prod_title = "Amazon Product"

    prod_img = (product_info.get('image_url') if product_info else '')
    if not prod_img and asin:
        prod_img = f"https://images-na.ssl-images-amazon.com/images/P/{asin}.01.LZZZZZZZ.jpg"

    prod_price = product_info.get('price', '') if product_info else ''
    prod_rating = product_info.get('rating_str', '') if product_info else ''
    prod_ratings_count = product_info.get('ratings_count', '') if product_info else ''
    prod_brand = product_info.get('brand', '') if product_info else ''
    prod_url = target_url

    genuine_pct = (df[df['Authenticity'] == 'Genuine'].shape[0] / len(df) * 100) if not df.empty else 0
    avg_polarity = df['Sentiment Polarity'].mean() if not df.empty else 0

    with st.container(border=True):
        col_img, col_details = st.columns([1, 4])
        with col_img:
            if prod_img:
                st.image(prod_img, use_container_width=True)
            else:
                st.markdown("<div style='font-size: 5rem; text-align: center;'>🛍️</div>", unsafe_allow_html=True)
        with col_details:
            badges = ["🛒 **Live Scraped from Amazon**"]
            if prod_brand:
                badges.append(f"🏷️ **{prod_brand}**")
            if prod_price:
                badges.append(f"💰 **{prod_price}**")
            if prod_rating:
                badges.append(f"⭐ **{prod_rating}**")
            if prod_ratings_count:
                badges.append(f"👥 {prod_ratings_count}")
            st.markdown(" &nbsp;|&nbsp; ".join(badges))
            
            st.subheader(prod_title)
            
            summary_items = [
                f"🛡️ **{genuine_pct:.1f}% Verified Authentic**",
                f"📊 **Avg Sentiment: {avg_polarity:.2f}**"
            ]
            if prod_url:
                summary_items.append(f"[🔗 View Product on Amazon ↗]({prod_url})")
            st.markdown(" &nbsp;•&nbsp; ".join(summary_items))


# ========== Enhanced Main App Functionality ==========
def main():
    # Theme toggle
    render_theme_toggle()
    
    # URL Input with Enhanced Validation
    with st.container():
        st.markdown("""
        <div style="position: relative;">
            <div class="blob" style="top: -50px; right: -50px; width: 200px; height: 200px;"></div>
            <div class="blob" style="bottom: -50px; left: -50px; width: 150px; height: 150px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1.2], vertical_alignment="bottom")
        with col1:
            url = st.text_input("Enter Amazon Product or Short URL:", placeholder="e.g. https://amzn.in/d/08obX5YK or https://www.amazon.in/dp/B0HGRM4S1B", key="url_input")
        with col2:
            analyze_btn = st.button("🚀 Analyze Reviews", key="analyze_btn", type="primary", use_container_width=True)

    # Input validation
    if analyze_btn:
        if not url:
            st.error("⚠️ Please enter an Amazon URL to analyze")
            return
        
        # Step 1: Resolve short link, redirects, and canonicalize URL
        is_valid, canonical_url, asin = resolve_and_canonicalize_amazon_url(url)
        if not is_valid:
            # Fallback to general validation
            is_valid_gen, error_msg = validate_amazon_url(url, config)
            if not is_valid_gen:
                st.error(f"❌ {error_msg}")
                return
        else:
            url = canonical_url

        if asin:
            st.info(f"🎯 **Target ASIN:** `{asin}` | **Canonical URL:** [{url}]({url})")
        
        # Sanitize input
        url = sanitize_input(url)
        st.session_state.last_analyzed_url = url
        
        # Check cache first
        cached_data = get_cached_data(url)
        if cached_data:
            st.info("📋 Using cached data (from previous analysis)")
            c_df = pd.DataFrame(cached_data['reviews'])
            if 'Cleaned_Review' not in c_df.columns:
                c_df, c_audit = DataCleaningPipeline().clean_reviews_dataframe(c_df)
                st.session_state.pipeline_audit = c_audit
            
            p_info = cached_data.get('product_info', {})
            if not p_info.get('image_url'):
                asin_m = re.search(r'/(?:dp|gp/product|product)/([A-Z0-9]{10})', url, re.I)
                if asin_m:
                    asin = asin_m.group(1).upper()
                    p_info['image_url'] = f"https://images-na.ssl-images-amazon.com/images/P/{asin}.01.LZZZZZZZ.jpg"
                    p_info['url'] = url
            st.session_state.product_info = p_info
            st.session_state.reviews_df = c_df
            st.session_state.analysis_done = True
            st.success(f"✅ Loaded {len(c_df)} reviews from cache!")
        else:
            with st.spinner("🌐 Fetching live reviews directly from Amazon..."):
                try:
                    # Step 1: Scrape real live reviews, product image, and metadata directly from Amazon
                    scrape_res = scraper.scrape_reviews_live(url)
                    if len(scrape_res) == 3:
                        real_product_name, reviews, product_info = scrape_res
                    else:
                        real_product_name, reviews = scrape_res
                        product_info = getattr(scraper, 'last_product_info', {})
                    st.session_state.product_info = product_info
                    
                    if not reviews:
                        # Fallback attempt to extract text from page body
                        full_text = scraper.extract_page_text(url)
                        if full_text:
                            review_section = extract_reviews(full_text)
                            if review_section:
                                for raw_rev in review_section.split("Report"):
                                    parsed = parse_review(raw_rev)
                                    if parsed:
                                        parsed['Product'] = real_product_name
                                        reviews.append(parsed)

                    if not reviews:
                        st.warning("⚠️ No customer reviews were found online for this product on Amazon. The product may be newly launched or have zero reviews yet.")
                        return

                    # Step 2: Documented Multi-Stage Data Cleaning Pipeline
                    cleaning_pipeline = DataCleaningPipeline()
                    raw_df = pd.DataFrame(reviews)
                    df, audit_metrics = cleaning_pipeline.clean_reviews_dataframe(raw_df)
                    st.session_state.pipeline_audit = audit_metrics
                    
                    # Step 3: High-Speed Unified NLP Analysis (Single-Pass Execution)
                    nlp_results = []
                    for _, row in df.iterrows():
                        res = nlp_analyzer.process_review_unified(row.to_dict())
                        nlp_results.append(res)
                    
                    res_df = pd.DataFrame(nlp_results, index=df.index)
                    for col in res_df.columns:
                        df[col] = res_df[col]
                    
                    # Ensure all expected columns are present
                    expected_columns = ['Product', 'Reviewer', 'Location', 'Date', 'Rating', 
                                      'Review', 'Cleaned_Review', 'Verified Purchase', 'Helpful Votes', 'Category',
                                      'Sentiment', 'Sentiment Analysis', 'Sentiment Polarity', 'Sentiment Subjectivity',
                                      'Authenticity', 'Authenticity Reason', 'Authenticity Score']
                    
                    for col in expected_columns:
                        if col not in df.columns:
                            df[col] = "N/A"
                    
                    # Assign true live product title
                    df['Product'] = real_product_name
                    
                    # Cache the results
                    cache_data(url, {
                        'reviews': df.to_dict('records'), 
                        'product_info': product_info,
                        'timestamp': datetime.now().isoformat()
                    }, config)
                    
                    st.session_state.reviews_df = df
                    st.session_state.analysis_done = True
                    st.success(f"✅ Successfully analyzed {len(df)} reviews!")
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.session_state.analysis_done = False

    if 'analysis_done' in st.session_state and st.session_state.analysis_done:
        df = st.session_state.reviews_df
        
        # Display Live Scraped Product Showcase Card
        product_info = st.session_state.get('product_info', {})
        render_product_showcase(product_info, df)

        # Summary Metrics with Animation
        cols = st.columns(4)
        metrics = [
            ("Total Reviews", len(df), "#6e8efb"),
            ("Average Rating", round(df['Rating'].mean(), 1), "#a777e3"),
            ("Authentic Reviews", f"{len(df[df['Authenticity'] == 'Genuine'])} ({df[df['Authenticity'] == 'Genuine'].shape[0]/len(df)*100:.1f}%)", "#00AA45"),
            ("Avg Sentiment", f"{df['Sentiment Polarity'].mean():.2f}", "#FFA500")
        ]

        
        for i, (title, value, color) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="animation-delay: {i*0.1}s;">
                    <h3 style="margin: 0 0 6px 0; font-size: 1.05rem;">{title}</h3>
                    <div class="metric-value" style="color: {color}; -webkit-text-fill-color: {color}; font-size: 2.1rem; font-weight: 800; margin: 4px 0;">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Rating Distribution
        st.markdown("## 📈 Rating Distribution")
        rating_dist = df['Rating'].value_counts().sort_index()
        fig_rating = px.bar(rating_dist, 
                           labels={'index': 'Star Rating', 'value': 'Count'},
                           color=rating_dist.index,
                           color_continuous_scale='Bluered')
        fig_rating.update_layout(showlegend=False)
        st.plotly_chart(fig_rating, use_container_width=True)
        
        # Enhanced Multi-Perspective Analytics Tabs
        st.markdown("## 🔍 Review Intelligence Suite")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🎯 Sentiment Accuracy & Evaluation", 
            "🏷️ Aspect Categories", 
            "🔬 Linguistic Patterns", 
            "🛠️ Data-Cleaning Pipeline", 
            "🔄 Product Comparison Dashboard"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                sentiment_dist = df['Sentiment'].value_counts()
                fig_sentiment = px.pie(sentiment_dist, 
                                      names=sentiment_dist.index,
                                      title="Review Sentiment Distribution",
                                      color=sentiment_dist.index,
                                      color_discrete_map={'Positive': '#00AA45', 'Negative': '#FF4D4D', 'Neutral': '#FFA500'})
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                auth_dist = df['Authenticity'].value_counts()
                fig_auth = px.pie(auth_dist, 
                                  names=auth_dist.index,
                                  title="Review Authenticity Distribution",
                                  color=auth_dist.index,
                                  color_discrete_map={'Genuine': '#00AA45', 'Likely Genuine': '#7FDBFF', 'Potentially Fake': '#FF4D4D'})
                st.plotly_chart(fig_auth, use_container_width=True)
                
            # Sentiment vs Authenticity
            st.markdown("### Sentiment vs Authenticity Correlation")
            fig_sent_auth = px.sunburst(df, path=['Authenticity', 'Sentiment'], 
                                       color='Sentiment',
                                       color_discrete_map={'Positive': '#00AA45', 'Negative': '#FF4D4D', 'Neutral': '#FFA500'})
            st.plotly_chart(fig_sent_auth, use_container_width=True)

        with tab2:
            st.markdown("### 🎯 Ground-Truth Sentiment Accuracy & Model Evaluation")
            st.markdown(
                "Benchmark NLP sentiment classifications against verified customer star ratings as proxy ground truth "
                "(**4-5 Stars** = Positive, **3 Stars** = Neutral, **1-2 Stars** = Negative)."
            )
            
            # Instant-load Session Cache for Evaluation
            prod_id = str(df['Product'].iloc[0]) if ('Product' in df and len(df) > 0) else "default_prod"
            if 'eval_cache' not in st.session_state or st.session_state.get('eval_cached_prod') != prod_id:
                eval_results = SentimentEvaluator.evaluate(df)
                st.session_state.eval_cache = eval_results
                st.session_state.eval_cached_prod = prod_id
                st.session_state.cm_fig = SentimentEvaluator.plot_confusion_matrix(eval_results['confusion_matrix'])
                st.session_state.perf_fig = SentimentEvaluator.plot_class_performance(eval_results['class_metrics'])
                st.session_state.scatter_fig = SentimentEvaluator.plot_rating_vs_polarity(df)
            
            eval_results = st.session_state.eval_cache
            
            # Key Evaluation Metric Cards
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                st.metric("Overall Accuracy", f"{eval_results['accuracy']}%", help="% of reviews where NLP sentiment matched star rating category")
            with ecol2:
                st.metric("Macro F1-Score", f"{eval_results['macro_f1']}", help="Unweighted average F1 across Positive, Neutral, Negative")
            with ecol3:
                st.metric("Normalized MAE", f"{eval_results['mae']}", help="Mean Absolute Error between normalized rating (0-1) and polarity (0-1)")
            with ecol4:
                st.metric("Rating/Polarity Correlation", f"r = {eval_results['correlation']}", help="Pearson correlation between rating and sentiment polarity")
            
            # Visual Charts: Confusion Matrix & Class Metrics
            cm_col, class_col = st.columns(2)
            with cm_col:
                st.plotly_chart(st.session_state.cm_fig, use_container_width=True)
            with class_col:
                st.plotly_chart(st.session_state.perf_fig, use_container_width=True)
            
            # Scatter Plot: Rating vs Polarity
            st.markdown("#### Star Rating vs Polarity Alignment")
            st.plotly_chart(st.session_state.scatter_fig, use_container_width=True)

            # Detailed Classification Table
            st.markdown("#### Class-by-Class Classification Report")
            class_report_data = []
            for c_name, c_vals in eval_results['class_metrics'].items():
                class_report_data.append({
                    "Class": c_name,
                    "Precision": c_vals['Precision'],
                    "Recall": c_vals['Recall'],
                    "F1-Score": c_vals['F1-Score'],
                    "Support": c_vals['Support'],
                    "True Positives (TP)": c_vals['TP'],
                    "False Positives (FP)": c_vals['FP'],
                    "False Negatives (FN)": c_vals['FN']
                })
            st.dataframe(pd.DataFrame(class_report_data), use_container_width=True)

            # Sarcasm & Divergence Inspector
            st.markdown("#### 🚨 High-Divergence Review Inspector (Sarcasm & Mixed Sentiment Detection)")
            discrepancies = eval_results.get('discrepancies', pd.DataFrame())
            if not discrepancies.empty:
                st.warning(f"Detected {len(discrepancies)} review(s) with severe star-rating vs sentiment divergence.")
                disp_cols = ['Reviewer', 'Rating', 'Ground_Truth', 'Sentiment', 'Sentiment Polarity', 'Discrepancy_Type', 'Review']
                st.dataframe(discrepancies[[c for c in disp_cols if c in discrepancies.columns]], use_container_width=True)
            else:
                st.success("✅ No extreme sentiment divergences (5-star negative or 1-star positive) detected in this review batch.")
        
        with tab3:
            # Category Distribution
            st.markdown("### Review Categories")
            category_dist = df['Category'].value_counts()
            fig_category = px.bar(category_dist, 
                                labels={'index': 'Category', 'value': 'Count'},
                                title="What Aspects Are Customers Talking About?")
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Category by Rating
            st.markdown("### Categories by Rating")
            fig_cat_rating = px.box(df, x='Category', y='Rating', 
                                  color='Category',
                                  title="How Do Ratings Vary by Category?")
            st.plotly_chart(fig_cat_rating, use_container_width=True)
        
        with tab4:
            fig_patterns = px.scatter(df, x='Rating', y='Authenticity Score', 
                                    color='Sentiment',
                                    size='Helpful Votes',
                                    color_discrete_map={'Positive': '#00AA45', 'Negative': '#FF4D4D', 'Neutral': '#FFA500'},
                                    hover_data=['Reviewer', 'Review', 'Category'],
                                    title="Rating vs Authenticity Score by Sentiment")
            st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Sentiment Polarity vs Subjectivity
            st.markdown("### Sentiment Analysis")
            fig_sent_polar = px.scatter(df, x='Sentiment Polarity', y='Sentiment Subjectivity',
                                       color='Authenticity',
                                       hover_data=['Review'],
                                       title="Sentiment Polarity vs Subjectivity")
            st.plotly_chart(fig_sent_polar, use_container_width=True)

            # Analyze linguistic features for a sample review
            st.markdown("### Linguistic Feature Analysis")
            stable_options = df['Review'].dropna().tolist()[:5]
            sample_review = st.selectbox("Select a review to analyze:", stable_options)
            
            if sample_review:
                if 'ling_cache' not in st.session_state:
                    st.session_state.ling_cache = {}
                if sample_review not in st.session_state.ling_cache:
                    st.session_state.ling_cache[sample_review] = nlp_analyzer.get_linguistic_features(sample_review)
                features = st.session_state.ling_cache[sample_review]
                
                # Display entities
                st.markdown("#### Named Entities")
                if features['entities']:
                    entity_df = pd.DataFrame(features['entities'], columns=["Entity", "Type"])
                    st.dataframe(entity_df)
                else:
                    st.info("No named entities found")
                
                # Display POS tags
                st.markdown("#### Part-of-Speech Tags")
                pos_df = pd.DataFrame(features['pos_tags'], columns=["Token", "POS", "Tag"])
                st.dataframe(pos_df)
                
                # Display noun chunks
                st.markdown("#### Noun Chunks")
                if features['noun_chunks']:
                    st.write(", ".join(features['noun_chunks']))
                else:
                    st.info("No noun chunks found")
                
                # Display statistics
                st.markdown("#### Text Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", features['word_count'])
                with col2:
                    st.metric("Sentence Count", features['sentence_count'])

        with tab5:
            st.markdown("### 🛠️ Documented Data-Cleaning Pipeline & Audit Trail")
            st.markdown(
                "A transparent, deterministic, multi-stage data preprocessing pipeline that purges platform boilerplate, "
                "standardizes linguistic contractions, and strips encoding noise to maximize NLP fidelity."
            )
            
            # Pipeline Audit Metrics
            audit = st.session_state.get('pipeline_audit', {})
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Raw Reviews Processed", audit.get('initial_review_count', len(df)))
            with p_col2:
                st.metric("Duplicate Reviews Culled", audit.get('duplicate_reviews_found', 0))
            with p_col3:
                st.metric("Overall Noise Reduction", f"{audit.get('overall_noise_reduction_pct', 0.0)}%")
            with p_col4:
                st.metric("Clean Tokens Available", audit.get('total_cleaned_characters', len(" ".join(df['Cleaned_Review']))))

            # Stage by Stage Documentation
            st.markdown(DataCleaningPipeline.get_pipeline_documentation_markdown())

            # Interactive Before & After Review Explorer
            st.markdown("#### 🔬 Interactive Before vs After Review Transformer Explorer")
            sample_idx = st.selectbox(
                "Select a review to inspect pipeline transformations:",
                range(len(df)),
                format_func=lambda i: f"Review #{i+1}: {df.iloc[i]['Review'][:60]}..."
            )
            
            selected_row = df.iloc[sample_idx]
            pipe_demo = DataCleaningPipeline().process_single_review(str(selected_row['Review']))
            
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("**Original Raw Scraped Text:**")
                st.text_area("Raw Review", value=pipe_demo['raw_text'], height=130, disabled=True, key=f"raw_{sample_idx}")
                st.caption(f"Raw Character Count: {pipe_demo['char_count_raw']}")
            with comp_col2:
                st.markdown("**Cleaned & Standardized Text:**")
                st.text_area("Cleaned Review", value=pipe_demo['cleaned_text'], height=130, disabled=True, key=f"clean_{sample_idx}")
                st.caption(f"Cleaned Characters: {pipe_demo['char_count_clean']} | Noise Culled: {pipe_demo['reduction_pct']}% | MD5: `{pipe_demo['hash'][:10]}...`")

            with st.expander("Show Sequential Transformation Stages"):
                st.markdown(f"**Stage 1 (Unicode/HTML Normalization):**\n```\n{pipe_demo['stage_1_normalized']}\n```")
                st.markdown(f"**Stage 2 (Boilerplate Stripped):**\n```\n{pipe_demo['stage_2_noise_stripped']}\n```")
                st.markdown(f"**Stage 3 (Contractions Expanded):**\n```\n{pipe_demo['stage_3_contractions_expanded']}\n```")
                st.markdown(f"**Stage 4 (Final Typography Cleaned):**\n```\n{pipe_demo['cleaned_text']}\n```")

        with tab6:
            st.markdown("### 🔄 Multi-Product Comparison Benchmark Dashboard")
            st.markdown("Benchmark the active product against competitors across customer satisfaction, sentiment tone, authenticity trust, and aspect-level dimensions.")

            # Active Product Summary
            active_product_name = df['Product'].iloc[0] if 'Product' in df and len(df) > 0 else "Active Product"
            active_summary = ProductComparator.compute_product_summary(active_product_name, df, url=url)
            
            if 'comparison_products' not in st.session_state:
                st.session_state.comparison_products = [active_summary]
            else:
                # Keep active product synchronized
                st.session_state.comparison_products[0] = active_summary

            # Controls to add more products
            comp_input_col, comp_btn_col, sample_btn_col = st.columns([3, 1, 1.2])
            with comp_input_col:
                new_comp_url = st.text_input("Enter competitor Amazon URL:", placeholder="https://www.amazon.in/dp/B09G9FPHY6", key="tab_comp_url")
            with comp_btn_col:
                add_url_clicked = st.button("Add Product", key="btn_add_comp_url")
            with sample_btn_col:
                add_sample_clicked = st.button("Add Sample Competitor", key="btn_add_sample_comp")

            if add_sample_clicked:
                # Add a realistic competitor sample for instant benchmarking
                sample_name = "Competitor Brand Pro X"
                if not any(p['name'] == sample_name for p in st.session_state.comparison_products):
                    # Create a realistic comparison DataFrame with slightly different metrics
                    comp_df = df.copy()
                    comp_df['Product'] = sample_name
                    # Shift ratings and polarity slightly to show contrast
                    comp_df['Rating'] = np.clip(comp_df['Rating'] + np.random.choice([-1, 0, 1], size=len(comp_df)), 1, 5)
                    comp_df['Sentiment Polarity'] = np.clip(comp_df['Sentiment Polarity'] * 0.85 + 0.05, -1.0, 1.0)
                    sample_summary = ProductComparator.compute_product_summary(sample_name, comp_df, url="https://amazon.com/sample-competitor")
                    st.session_state.comparison_products.append(sample_summary)
                    st.success(f"Added '{sample_name}' to comparison dashboard!")
                    st.rerun()

            if add_url_clicked and new_comp_url:
                is_valid, msg = validate_amazon_url(new_comp_url, config)
                if is_valid:
                    with st.spinner("Scraping and analyzing competitor reviews..."):
                        c_text = scraper.extract_page_text(new_comp_url)
                        c_rev_sec = extract_reviews(c_text)
                        c_raw_reviews = []
                        c_current = []
                        for l in c_rev_sec.split('\n'):
                            l = l.strip()
                            if l.startswith("Report") and c_current:
                                c_raw_reviews.append("\n".join(c_current))
                                c_current = []
                            elif l:
                                c_current.append(l)
                        c_parsed = [parse_review(r) for r in c_raw_reviews if parse_review(r)]
                        if c_parsed:
                            c_new_df, _ = DataCleaningPipeline().clean_reviews_dataframe(pd.DataFrame(c_parsed))
                            # Quick NLP pass
                            c_new_df['Sentiment'] = [nlp_analyzer.analyze_sentiment(t)[0] for t in c_new_df['Cleaned_Review']]
                            c_new_df['Sentiment Polarity'] = [nlp_analyzer.analyze_sentiment(t)[2] for t in c_new_df['Cleaned_Review']]
                            c_new_df['Category'] = [nlp_analyzer.categorize_review_content(t) for t in c_new_df['Cleaned_Review']]
                            c_new_df['Authenticity'] = [nlp_analyzer.check_authenticity(r)[0] for _, r in c_new_df.iterrows()]
                            c_new_df['Authenticity Score'] = [nlp_analyzer.check_authenticity(r)[2] for _, r in c_new_df.iterrows()]
                            
                            p_name = new_comp_url.split('/')[-1].split('?')[0] if len(new_comp_url.split('/')) > 3 else "Competitor"
                            c_new_df['Product'] = p_name
                            new_summary = ProductComparator.compute_product_summary(p_name, c_new_df, url=new_comp_url)
                            st.session_state.comparison_products.append(new_summary)
                            st.success(f"Added '{p_name}' to comparison dashboard!")
                            st.rerun()
                else:
                    st.error(msg)

            # Render Multi-Product Benchmark Visuals
            prods = st.session_state.comparison_products
            if len(prods) >= 1:
                # Comparative KPI Table
                st.markdown("#### 📊 Comparative Metric Matrix")
                comp_rows = []
                for p in prods:
                    comp_rows.append({
                        "Product": p['name'],
                        "Total Reviews": p['total_reviews'],
                        "Avg Rating (/5)": p['avg_rating'],
                        "Sentiment Polarity": p['avg_polarity'],
                        "Positive Reviews %": f"{p['positive_pct']}%",
                        "Authentic Reviews %": f"{p['genuine_pct']}%",
                        "Authenticity Index (/100)": p['avg_authenticity_score']
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

                # Radar and Grouped Bar Charts
                col_radar, col_bars = st.columns(2)
                with col_radar:
                    radar_fig = ProductComparator.plot_aspect_radar(prods)
                    st.plotly_chart(radar_fig, use_container_width=True)
                with col_bars:
                    bar_fig = ProductComparator.plot_metric_comparison_bars(prods)
                    st.plotly_chart(bar_fig, use_container_width=True)

                # Automated Verdict Box
                verdicts = ProductComparator.generate_recommendation_verdict(prods)
                if verdicts:
                    st.markdown("#### 🏆 Algorithmic Recommendation & Market Verdict")
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.info(verdicts.get('top_rated', ''))
                        st.success(verdicts.get('top_sentiment', ''))
                    with v_col2:
                        st.warning(verdicts.get('highest_trust', ''))
                        st.info(verdicts.get('value_champion', ''))

                if st.button("Reset Comparison List", key="reset_comp"):
                    st.session_state.comparison_products = [active_summary]
                    st.rerun()

        # Highlighted Reviews
        st.markdown("## 🔎 Highlighted Reviews")
        sample_reviews = df.sample(min(5, len(df)))
        
        for _, row in sample_reviews.iterrows():
            sentiment_class = "review-genuine" if row['Authenticity'] == "Genuine" else "review-fake"
            tag_class = "tag-genuine" if row['Authenticity'] == "Genuine" else "tag-fake"
            
            st.markdown(f"""
            <div class="review-card {sentiment_class}" style="animation-delay: 0.3s;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span class="review-tag {tag_class}">{row['Authenticity']}</span>
                        <span class="review-tag" style="background: #E6F7EE; color: #6e8efb;">{row['Category']}</span>
                        <span style="font-weight: 600; color: #2a3f5f;">{row['Reviewer']}</span>
                        <span style="color: #666; font-size: 0.9rem;">from {row['Location']}</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        {"★" * int(row['Rating'])}{"☆" * (5 - int(row['Rating']))}
                        <span style="margin-left: 10px; font-weight: 600; color: #2a3f5f;">{row['Rating']}/5</span>
                    </div>
                </div>
                <p style="margin: 10px 0; font-size: 1rem; line-height: 1.6;">"{row['Review']}"</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #666; font-size: 0.9rem;">{row['Date']} • {row['Verified Purchase']} Verified</span>
                    <span style="color: #666; font-size: 0.9rem;">{row['Helpful Votes']} helpful votes</span>
                </div>
                <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.03); border-radius: 8px;">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Sentiment:</strong> {row['Sentiment']} (Polarity: {row['Sentiment Polarity']:.2f}, Subjectivity: {row['Sentiment Subjectivity']:.2f})</p>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9rem;"><strong>Authenticity:</strong> Score: {row['Authenticity Score']}/100 - {row['Authenticity Reason']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Time Series Analysis
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            st.markdown("## ⏳ Review Activity Over Time")
            time_series = df.groupby(['Date', 'Authenticity']).size().unstack().fillna(0)
            fig_time = px.line(time_series, 
                              labels={'value': 'Number of Reviews', 'Date': 'Date'},
                              color_discrete_map={'Genuine': '#00AA45', 'Likely Genuine': '#7FDBFF', 'Potentially Fake': '#FF4D4D'})
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Full Data Table
        st.markdown("## 📋 Complete Review Data")
        st.dataframe(df, use_container_width=True)

        
        # Export Section
        st.markdown("## 💾 Export Analysis Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="amazon_review_analysis.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        with col2:
            # For Excel download, we need to save to a buffer first
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name="amazon_review_analysis.xlsx",
                mime="application/vnd.ms-excel",
                key="download_excel"
            )
        
        with col3:
            # JSON Export
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name="amazon_review_analysis.json",
                mime="application/json",
                key="download_json"
            )
        
        with col4:
            # Generate comprehensive report
            if st.button("Generate Report", key="generate_report"):
                generate_comprehensive_report(df)

def generate_comprehensive_report(df):
    """Generate and display an executive intelligence report from analyzed reviews"""
    total = len(df)
    if total == 0:
        st.warning("No data available to generate report.")
        return

    avg_rating = df['Rating'].mean() if 'Rating' in df else 0
    genuine_count = len(df[df['Authenticity'] == 'Genuine']) if 'Authenticity' in df else 0
    genuine_pct = (genuine_count / total * 100) if total else 0
    pos_count = len(df[df['Sentiment'] == 'Positive']) if 'Sentiment' in df else 0
    pos_pct = (pos_count / total * 100) if total else 0

    st.markdown("### 📑 Executive Review Intelligence Report")
    report_text = f"""# Executive Review Intelligence Report
**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analyzed Product:** {df['Product'].iloc[0] if 'Product' in df else 'Amazon Product'}

## Key Performance Indicators
- **Total Reviews Evaluated:** {total}
- **Average Customer Rating:** {avg_rating:.2f} / 5.0
- **Authenticity Confidence:** {genuine_pct:.1f}% Genuine reviews ({genuine_count}/{total})
- **Positive Sentiment Share:** {pos_pct:.1f}% ({pos_count}/{total})

## Sentiment Summary
- Positive Reviews: {len(df[df['Sentiment'] == 'Positive']) if 'Sentiment' in df else 0}
- Neutral Reviews: {len(df[df['Sentiment'] == 'Neutral']) if 'Sentiment' in df else 0}
- Negative Reviews: {len(df[df['Sentiment'] == 'Negative']) if 'Sentiment' in df else 0}

## Authenticity Audit
- Genuine: {genuine_count}
- Likely Genuine: {len(df[df['Authenticity'] == 'Likely Genuine']) if 'Authenticity' in df else 0}
- Potentially Fake / Suspicious: {len(df[df['Authenticity'] == 'Potentially Fake']) if 'Authenticity' in df else 0}

## Top Aspect Categories
{df['Category'].value_counts().to_string() if 'Category' in df else 'N/A'}
"""
    st.info(f"**Report Generated Successfully** for {total} customer reviews.")
    st.markdown(f"```markdown\n{report_text}\n```")
    st.download_button(
        label="📥 Download Intelligence Report (.md)",
        data=report_text,
        file_name="executive_review_report.md",
        mime="text/markdown",
        key="download_report_md"
    )



# ========== Sidebar with Enhanced UI ==========
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="position: relative; display: inline-block;">
                <div class="blob" style="top: -20px; left: -20px; width: 100px; height: 100px;"></div>
                <svg width="100" height="100" viewBox="0 0 100 100" style="position: relative; z-index: 1;">
                    <circle cx="50" cy="50" r="45" fill="#6E8EFB"/>
                    <path d="M70 40L45 65L30 50" stroke="white" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                </svg>
            </div>
            <h3 style="margin-top: 15px;">TruthLens AI</h3>
            <p style="color: #666; font-size: 0.9rem;">Amazon Review & Authenticity Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔍 How It Works")
        st.markdown("""
        <div class="timeline">
            <div class="timeline-item">
                <h4 style="margin: 0 0 5px 0;">1. Paste Product URL</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Copy any Amazon product link and paste it above</p>
            </div>
            <div class="timeline-item">
                <h4 style="margin: 0 0 5px 0;">2. Analyze Reviews</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Our NLP models process all customer reviews</p>
            </div>
            <div class="timeline-item">
                <h4 style="margin: 0 0 5px 0;">3. Get Insights</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Discover sentiment, authenticity, and key topics</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 NLP Features")
        st.markdown("""
        <div class="accordion">
            <div class="accordion-item">
                <div class="accordion-header">Sentiment Analysis</div>
                <div class="accordion-content">
                    Uses TextBlob to determine polarity (-1 to 1) and subjectivity (0 to 1) of each review.
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">Content Categorization</div>
                <div class="accordion-content">
                    Identifies key aspects like Quality, Performance, Design, Value, etc. using spaCy.
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">Authenticity Detection</div>
                <div class="accordion-content">
                    Analyzes linguistic patterns to detect potentially fake reviews.
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">Linguistic Features</div>
                <div class="accordion-content">
                    Examines POS tags, named entities, and dependency trees for deeper analysis.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔔 Tips for Best Results")
        st.markdown("""
        - Analyze products with at least 20 reviews
        - Compare with similar products
        - Look for verified purchase reviews
        - Check review dates for patterns
        - Examine the linguistic features tab for deeper insights
        """)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>This tool provides estimates based on NLP analysis.</p>
            <p>Results should be considered as one of several factors.</p>
            <p style="margin-top: 20px;">© 2026 TruthLens AI</p>
        </div>
        """, unsafe_allow_html=True)

# ========== Run the App ==========
if __name__ == "__main__":
    render_sidebar()
    main()
    
    # Add some floating elements for visual appeal
    st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: -1;">
        <div class="blob floating" style="width: 150px; height: 150px; background: rgba(110,142,251,0.1);"></div>
    </div>
    <div style="position: fixed; top: 100px; left: -50px; z-index: -1;">
        <div class="blob floating" style="width: 200px; height: 200px; background: rgba(167,119,227,0.1); animation-delay: 2s;"></div>
    </div>
    """, unsafe_allow_html=True)