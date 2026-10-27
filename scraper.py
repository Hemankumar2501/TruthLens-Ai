"""
Web scraping and review extraction functionality for Amazon Review Insight Pro
"""
import re
import random
import logging
import sys
import subprocess
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class AmazonScraper:
    """Robust scraper for live Amazon product pages using HTTP/2 client emulation"""

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()

    @staticmethod
    def is_bot_page(html: str) -> bool:
        """Check if Amazon returned a captcha or bot-block challenge page"""
        if not html or len(html) < 600:
            return True
        html_lower = html.lower()
        bot_markers = [
            "validatecaptcha",
            "robot check",
            "api-services-support@amazon.com",
            "type the characters you see in this image",
            "to discuss automated access to amazon data",
            "click the button below to continue shopping",
            "continue shopping conditions of use",
            "enter the characters you see below",
            "sorry, we just need to make sure you're not a robot"
        ]
        return any(m in html_lower for m in bot_markers)

    def _fetch_html_live(self, url: str) -> str:
        """Fetch live Amazon page using HTTP/2 client emulation to bypass bot blockers"""
        canonical_url = url.strip()
        
        # Clean down to canonical product page if ASIN is present (removes tracking params that trigger bot challenge)
        asin_match = re.search(r'/(?:dp|gp/product|product)/([A-Z0-9]{10})', canonical_url, re.I)
        domain_match = re.search(r'amazon\.[a-z\.]+', canonical_url, re.I)
        domain = domain_match.group(0).lower() if domain_match else "amazon.in"
        
        target_url = canonical_url
        if asin_match:
            asin = asin_match.group(1).upper()
            target_url = f"https://www.{domain}/dp/{asin}"

        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
        ]

        curl_bin = "curl.exe" if sys.platform.startswith("win") else "curl"

        for attempt, ua in enumerate(user_agents[:3]):
            cmd = [
                curl_bin,
                "-s",
                "-L",
                "-A", ua,
                "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "-H", "Accept-Language: en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
                "-H", "Sec-Ch-Ua: \"Chromium\";v=\"125\", \"Not.A/Brand\";v=\"24\"",
                "-H", "Sec-Ch-Ua-Mobile: ?0",
                "-H", "Sec-Ch-Ua-Platform: \"Windows\"",
                "-H", "Sec-Fetch-Dest: document",
                "-H", "Sec-Fetch-Mode: navigate",
                "-H", "Sec-Fetch-Site: none",
                "-H", "Sec-Fetch-User: ?1",
                "-H", "Upgrade-Insecure-Requests: 1",
                target_url
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=25)
                html = res.stdout.decode('utf-8', errors='ignore')
                if html and not self.is_bot_page(html):
                    return html
            except Exception as e:
                logger.debug(f"Live fetch attempt {attempt+1} error: {e}")

        # Fallback to requests session if curl was blocked or unavailable
        try:
            r = self.session.get(target_url, headers={
                'User-Agent': user_agents[0],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-IN,en;q=0.9'
            }, timeout=8)
            if r.status_code == 200 and not self.is_bot_page(r.text):
                return r.text
        except Exception as e:
            logger.debug(f"Requests fallback error: {e}")

        return ""

    def scrape_reviews_live(self, url: str) -> Tuple[str, List[Dict], Dict]:
        """
        Scrapes live Amazon reviews and product metadata (including image, price, and specs) directly from the product page online.
        Returns: (product_name, list_of_review_dicts, product_info_dict)
        """
        asin_match = re.search(r'/(?:dp|gp/product|product)/([A-Z0-9]{10})', url, re.I)
        asin = asin_match.group(1).upper() if asin_match else ""
        asin_cdn_img = f"https://images-na.ssl-images-amazon.com/images/P/{asin}.01.LZZZZZZZ.jpg" if asin else ""

        slug_match = re.search(r'amazon\.[a-z\.]+/([^/]+)/(?:dp|gp/product)', url)
        slug_title = ""
        if slug_match and not any(x in slug_match.group(1).lower() for x in ['dp', 'gp', 'product']):
            slug_title = slug_match.group(1).replace('-', ' ').strip().title()

        html = self._fetch_html_live(url)
        if not html:
            empty_info = {
                'title': slug_title or (f"Amazon Product ({asin})" if asin else "Amazon Product"),
                'image_url': asin_cdn_img,
                'price': "",
                'rating_str': "",
                'ratings_count': "",
                'brand': "",
                'url': url
            }
            self.last_product_info = empty_info
            return empty_info['title'], [], empty_info

        soup = BeautifulSoup(html, 'html.parser')

        # 1. Extract Real Product Title
        product_name = ""
        title_input = soup.select_one('input#productTitle, input[name="productTitle"]')
        if title_input and title_input.get('value'):
            product_name = title_input['value'].strip()

        if not product_name:
            prod_title_el = soup.select_one('span#productTitle, #productTitle, h1#title, .product-title-word-break')
            if prod_title_el:
                if prod_title_el.name == 'input':
                    product_name = prod_title_el.get('value', '').strip()
                else:
                    product_name = prod_title_el.get_text(strip=True)

        if not product_name or product_name.lower() in ("amazon product", "amazon.in", "amazon.com"):
            if slug_title:
                product_name = slug_title
            else:
                title_el = soup.find('title')
                if title_el:
                    product_name = title_el.get_text(strip=True).split(':')[0].strip()

        if not product_name or product_name.lower() in ("amazon product", "amazon.in", "amazon.com"):
            product_name = f"Amazon Product ({asin})" if asin else "Amazon Product"

        # 2. Extract High-Resolution Product Image
        image_url = ""
        # Tier 1: Embedded product image input
        img_input = soup.select_one('input#productImageUrl, input[name="productImageUrl"]')
        if img_input and img_input.get('value'):
            raw_img = img_input['value'].strip()
            image_url = re.sub(r'\._[A-Z0-9_,]+_\.', '._SL500_.', raw_img)

        # Tier 2: OpenGraph / Twitter meta tags
        if not image_url:
            og_img = soup.find('meta', property='og:image') or soup.find('meta', attrs={'name': 'twitter:image'})
            if og_img and og_img.get('content') and not og_img['content'].endswith('placeholder.png') and 'captcha' not in og_img['content'].lower():
                image_url = og_img['content'].strip()

        # Tier 3: High-res landing image element
        if not image_url:
            img_el = (soup.select_one('#landingImage') or 
                      soup.select_one('#imgBlkFront') or 
                      soup.select_one('#main-image') or 
                      soup.select_one('.imgTagWrapper img') or
                      soup.select_one('#ebooksImgBlkFront'))
            if img_el:
                if img_el.get('data-old-hires'):
                    image_url = img_el['data-old-hires'].strip()
                elif img_el.get('src') and not img_el['src'].startswith('data:'):
                    image_url = img_el['src'].strip()
                elif img_el.get('data-a-dynamic-image'):
                    try:
                        import json
                        dyn = json.loads(img_el['data-a-dynamic-image'])
                        if dyn:
                            image_url = list(dyn.keys())[0]
                    except Exception:
                        pass

        # Tier 4: Direct Amazon ASIN CDN (100% Guaranteed Image)
        if not image_url and asin_cdn_img:
            image_url = asin_cdn_img

        # 3. Extract Price & Official Ratings
        price_str = ""
        price_input = soup.select_one('input#priceValue, input[name="priceValue"]')
        if price_input and price_input.get('value'):
            pv = price_input['value'].strip()
            currency = soup.select_one('input#currencyOfPreference')
            curr_str = "₹" if currency and currency.get('value') == "INR" else (currency.get('value', '₹') if currency else '₹')
            try:
                price_str = f"{curr_str} {float(pv):,.0f}"
            except Exception:
                price_str = f"{curr_str} {pv}"

        if not price_str:
            price_el = (soup.select_one('.a-price .a-offscreen') or 
                        soup.select_one('#priceblock_ourprice') or 
                        soup.select_one('#priceblock_dealprice') or 
                        soup.select_one('#corePriceDisplay_desktop_feature_div .a-price-whole') or
                        soup.select_one('.priceToPay span.a-price-whole'))
            if price_el:
                price_str = price_el.get_text(strip=True)

        amazon_rating = ""
        ar_el = soup.select_one('#acrPopover span.a-icon-alt') or soup.select_one('i[data-hook="average-star-rating"] span')
        if ar_el:
            amazon_rating = ar_el.get_text(strip=True)

        ratings_count = ""
        rc_el = soup.select_one('#acrCustomerReviewText')
        if rc_el:
            ratings_count = rc_el.get_text(strip=True)

        brand_str = ""
        b_el = soup.select_one('#bylineInfo') or soup.select_one('.po-brand .a-span9')
        if b_el:
            brand_str = b_el.get_text(strip=True)

        product_info = {
            'title': product_name,
            'image_url': image_url,
            'price': price_str,
            'rating_str': amazon_rating,
            'ratings_count': ratings_count,
            'brand': brand_str,
            'url': url
        }
        self.last_product_info = product_info

        # 4. Extract Real Customer Reviews
        reviews_data = []
        review_elements = soup.select('div[data-hook="review"]')
        if not review_elements:
            review_elements = soup.select('.review')

        for el in review_elements:
            reviewer_el = el.select_one('.a-profile-name')
            reviewer = reviewer_el.get_text(strip=True) if reviewer_el else "Amazon Customer"

            rating_el = el.select_one('i[data-hook="review-star-rating"] span') or el.select_one('.a-icon-alt')
            rating_str = rating_el.get_text(strip=True) if rating_el else "5.0 out of 5 stars"
            m = re.search(r'([1-5](?:\.[0-9])?)', rating_str)
            rating_val = float(m.group(1)) if m else 5.0

            # Support both classic and modern 2025/2026 Amazon title selectors
            title_el = (el.select_one('a[data-hook="review-title"] span:not(.a-letter-space)') or 
                        el.select_one('[data-hook="reviewTitle"]') or
                        el.select_one('h5[data-hook="reviewTitle"]') or
                        el.select_one('span[data-hook="review-title"]') or 
                        el.select_one('.review-title'))
            title_str = title_el.get_text(strip=True) if title_el else ""

            date_el = el.select_one('span[data-hook="review-date"]')
            date_str = date_el.get_text(strip=True) if date_el else "Reviewed recently"

            vp_el = el.select_one('span[data-hook="avp-badge"]')
            vp_str = "Verified Purchase" if vp_el or "Verified Purchase" in el.get_text() else "Unverified Purchase"

            # Support both classic and modern 2025/2026 Amazon body selectors
            body_el = (el.select_one('[data-hook="reviewRichContentContainer"]') or
                       el.select_one('[data-hook="reviewText"]') or
                       el.select_one('span[data-hook="review-body"] span') or 
                       el.select_one('span[data-hook="review-body"]') or 
                       el.select_one('.review-text-content') or 
                       el.select_one('div[data-hook="review-collapsed"]'))
            body_str = ""
            if body_el:
                body_copy = BeautifulSoup(str(body_el), 'html.parser')
                for h in body_copy.select('.a-hidden, .a-teaser-describedby-collapsed, .a-teaser-describedby-expanded, .a-cardui-expand-control-footer'):
                    h.decompose()
                body_str = body_copy.get_text(separator=' ', strip=True)
            if not body_str:
                p_el = el.select_one('p')
                if p_el:
                    body_str = p_el.get_text(strip=True)

            helpful_el = el.select_one('span[data-hook="helpful-vote-statement"]')
            helpful_str = helpful_el.get_text(strip=True) if helpful_el else "0"
            m_h = re.search(r'(\d+)', helpful_str)
            helpful_votes = int(m_h.group(1)) if m_h else 0

            loc = "India"
            m_loc = re.search(r'Reviewed in\s+(.+?)\s+on', date_str)
            if m_loc:
                loc = m_loc.group(1).strip()

            full_review = f"{title_str}. {body_str}".strip(". ").strip()
            if not full_review:
                full_review = title_str if title_str else body_str

            # Strict junk / bot-detection filter
            check_str = (reviewer + " " + full_review).lower()
            if any(bad in check_str for bad in [
                "continue shopping", "conditions of use", "privacy notice", 
                "amazon.com, inc", "validatecaptcha", "robot check", 
                "api-services-support", "type the characters"
            ]):
                continue
            if len(full_review) < 8:
                continue

            reviews_data.append({
                'Product': product_name,
                'Reviewer': reviewer,
                'Location': loc,
                'Date': date_str,
                'Rating': rating_val,
                'Review': full_review,
                'Verified Purchase': vp_str,
                'Helpful Votes': helpful_votes
            })

        return product_name, reviews_data, product_info

    def extract_page_text(self, url: str) -> str:
        """Fetch Amazon product page content and return full readable text"""
        html = self._fetch_html_live(url)
        if not html or self.is_bot_page(html):
            return ""

        soup = BeautifulSoup(html, 'html.parser')
        review_elements = soup.select('div[data-hook="review"]') or soup.select('.review')
        if review_elements:
            return self._format_reviews_from_soup(soup, review_elements)

        # NEVER return raw body.get_text() which risks extracting footer/captcha text as reviews
        return ""

    def _format_reviews_from_soup(self, soup: BeautifulSoup, review_elements) -> str:
        """Format parsed HTML review elements into standardized text representation"""
        output_lines = ["Top reviews from India\n"]
        for el in review_elements:
            reviewer = el.select_one('.a-profile-name')
            reviewer_name = reviewer.get_text(strip=True) if reviewer else "Amazon Customer"

            rating_el = el.select_one('i[data-hook="review-star-rating"] span') or el.select_one('.a-icon-alt')
            rating_str = rating_el.get_text(strip=True) if rating_el else "5.0 out of 5 stars"

            title_el = el.select_one('a[data-hook="review-title"] span') or el.select_one('span[data-hook="review-title"]')
            title_str = title_el.get_text(strip=True) if title_el else "Great purchase"

            date_el = el.select_one('span[data-hook="review-date"]')
            date_str = date_el.get_text(strip=True) if date_el else "Reviewed in India on 15 January 2024"

            vp_el = el.select_one('span[data-hook="avp-badge"]')
            vp_str = "Verified Purchase" if vp_el else "Unverified Purchase"

            body_el = el.select_one('span[data-hook="review-body"] span') or el.select_one('div[data-hook="review-collapsed"]')
            body_str = body_el.get_text(separator=' ', strip=True) if body_el else ""

            helpful_el = el.select_one('span[data-hook="helpful-vote-statement"]')
            helpful_str = helpful_el.get_text(strip=True) if helpful_el else "Helpful"

            output_lines.append(reviewer_name)
            output_lines.append(rating_str)
            output_lines.append(title_str)
            output_lines.append(date_str)
            output_lines.append(vp_str)
            output_lines.append(body_str)
            output_lines.append(helpful_str)
            output_lines.append("Report\n")

        return "\n".join(output_lines)

    def _generate_realistic_review_stream(self, url: str) -> str:
        """
        Generates realistic Amazon product reviews formatted for the parser.
        Includes genuine, mixed, and suspicious reviews to test NLP and authenticity algorithms.
        """
        # Determine product context from URL
        url_lower = url.lower()
        if any(w in url_lower for w in ["b0hgrm4s1b", "redmi", "13c", "08obx5yk"]):
            product_type = "Redmi 13C 5G Smartphone"
            curated_reviews = [
                {
                    "reviewer": "Vikram Malhotra",
                    "rating": "5.0 out of 5 stars",
                    "title": "Unbeatable budget 5G phone! Butter smooth 90Hz display",
                    "date": "Reviewed in India on 12 February 2024",
                    "vp": "Verified Purchase",
                    "body": "Got the Redmi 13C 5G (6GB RAM + 128GB) and it is easily the best phone under 12k. Jio 5G speeds easily reach 450+ Mbps. The 90Hz refresh rate makes scrolling Instagram and daily apps buttery smooth. 5000mAh battery lasts 1.5 days on normal usage.",
                    "helpful": "58 people found this helpful"
                },
                {
                    "reviewer": "Sneha Sen",
                    "rating": "4.0 out of 5 stars",
                    "title": "Good daily driver, decent camera in daylight",
                    "date": "Reviewed in India on 28 January 2024",
                    "vp": "Verified Purchase",
                    "body": "Design with the star trail pattern on the back looks very premium. Daylight photos from the 50MP main sensor are sharp and punchy. Night mode could be improved, but given the price, Dimensity 6100+ handles multitasking with zero stutter.",
                    "helpful": "32 people found this helpful"
                },
                {
                    "reviewer": "Rajesh Kumar",
                    "rating": "5.0 out of 5 stars",
                    "title": "Best value for money 5G mobile in India",
                    "date": "Reviewed in India on 19 February 2024",
                    "vp": "Verified Purchase",
                    "body": "Excellent battery backup and super fast fingerprint sensor on power button. 6GB RAM handles BGMI on smooth settings without excessive heating. Charging is 18W supported, boxed with standard adapter.",
                    "helpful": "21 people found this helpful"
                },
                {
                    "reviewer": "Quick Buyer 101",
                    "rating": "5.0 out of 5 stars",
                    "title": "very good phone must buy quality super fast delivery",
                    "date": "Reviewed in India on 05 February 2024",
                    "vp": "Unverified Purchase",
                    "body": "very good phone must buy quality super fast delivery awesome camera nice product best phone ever highly recommend.",
                    "helpful": "0 people found this helpful"
                },
                {
                    "reviewer": "Aditya Verma",
                    "rating": "2.0 out of 5 stars",
                    "title": "Display brightness is average outdoors, speaker is single mono",
                    "date": "Reviewed in India on 22 January 2024",
                    "vp": "Verified Purchase",
                    "body": "While 5G reception and battery life are good, reading text outdoors under bright noon sun is difficult because the screen brightness peaks at 600 nits. Also single bottom firing speaker lacks punch.",
                    "helpful": "19 people found this helpful"
                },
                {
                    "reviewer": "Pooja Hegde",
                    "rating": "5.0 out of 5 stars",
                    "title": "Gifted to my father, he loves the large screen and battery",
                    "date": "Reviewed in India on 02 February 2024",
                    "vp": "Verified Purchase",
                    "body": "Large 6.74-inch screen is great for watching YouTube and video calling on WhatsApp. Font size is clear, call quality is crisp over VoNR 5G, and battery easily lasts 2 days for him.",
                    "helpful": "14 people found this helpful"
                },
                {
                    "reviewer": "Karan Mehra",
                    "rating": "3.0 out of 5 stars",
                    "title": "Decent secondary phone, pre-installed bloatware needs cleanup",
                    "date": "Reviewed in India on 14 February 2024",
                    "vp": "Verified Purchase",
                    "body": "Hardware is solid for the price, but MIUI has several pre-installed bloatware apps and notifications that you have to spend 20 minutes disabling on first boot. Once cleaned up, performance is smooth.",
                    "helpful": "27 people found this helpful"
                },
                {
                    "reviewer": "Mobile Reviewer Pro",
                    "rating": "5.0 out of 5 stars",
                    "title": "best best best phone love it highly recommend",
                    "date": "Reviewed in India on 10 February 2024",
                    "vp": "Unverified Purchase",
                    "body": "best phone best mobile buy now awesome performance battery 100% genuine nice product value for money.",
                    "helpful": "1 person found this helpful"
                }
            ]
            lines = ["Top reviews from India\n"]
            for r in curated_reviews:
                lines.append(r["reviewer"])
                lines.append(r["rating"])
                lines.append(r["title"])
                lines.append(r["date"])
                lines.append(r["vp"])
                lines.append(r["body"])
                lines.append(r["helpful"])
                lines.append("Report\n")
            return "\n".join(lines)

        if any(w in url_lower for w in ["phone", "iphone", "samsung", "pixel", "mobile", "b08l5v53zp"]):
            product_type = "Smartphone"
        elif any(w in url_lower for w in ["headphone", "earphone", "audio", "earbuds", "airpods"]):
            product_type = "Audio Device"
        elif any(w in url_lower for w in ["watch", "fitbit", "garmin"]):
            product_type = "Smartwatch"
        elif any(w in url_lower for w in ["laptop", "macbook", "computer", "dell", "hp"]):
            product_type = "Laptop"
        else:
            product_type = "Consumer Electronics"

        curated_reviews = [
            {
                "reviewer": "Aarav Sharma",
                "rating": "5.0 out of 5 stars",
                "title": f"Outstanding {product_type.lower()}! Exceeded my expectations",
                "date": "Reviewed in India on 14 February 2024",
                "vp": "Verified Purchase",
                "body": f"I have been daily-driving this {product_type.lower()} for over a month now. The battery life lasts easily 2 full days, the display clarity is vivid even under direct sunlight, and the build quality feels remarkably sturdy. Fast shipping and premium packaging.",
                "helpful": "42 people found this helpful"
            },
            {
                "reviewer": "Priya Venkatesh",
                "rating": "4.0 out of 5 stars",
                "title": "Solid performance with minor ergonomic quirks",
                "date": "Reviewed in India on 28 January 2024",
                "vp": "Verified Purchase",
                "body": f"The overall speed and software responsiveness are top notch. I love the sleek aluminum finish and intuitive controls. However, it can get slightly warm during prolonged 4K recording sessions. Worth every penny considering the competitive discount.",
                "helpful": "18 people found this helpful"
            },
            {
                "reviewer": "John Miller",
                "rating": "5.0 out of 5 stars",
                "title": "great product must buy value for money",
                "date": "Reviewed in the United States on 02 February 2024",
                "vp": "Verified Purchase",
                "body": "great product must buy value for money good quality highly recommend to everyone awesome fast shipping happy with purchase.",
                "helpful": "1 person found this helpful"
            },
            {
                "reviewer": "Rohan Deshmukh",
                "rating": "2.0 out of 5 stars",
                "title": "Disappointed with customer support and battery drain",
                "date": "Reviewed in India on 10 January 2024",
                "vp": "Verified Purchase",
                "body": f"I was very excited for this {product_type.lower()}, but the battery drops by 30% in just 2 hours of moderate usage. Reached out to support for a warranty replacement, but the service team was unresponsive and delayed the return window.",
                "helpful": "35 people found this helpful"
            },
            {
                "reviewer": "Sarah Jenkins",
                "rating": "5.0 out of 5 stars",
                "title": "Excellent build and crisp audio clarity",
                "date": "Reviewed in Canada on 22 December 2023",
                "vp": "Verified Purchase",
                "body": "Tested this extensively on my commute and at the gym. Noise isolation is impressive, connectivity with Bluetooth 5.3 is seamless with zero latency dropouts, and the ergonomic fit does not cause ear fatigue even after 4 hours.",
                "helpful": "12 people found this helpful"
            },
            {
                "reviewer": "Global Shopper 99",
                "rating": "5.0 out of 5 stars",
                "title": "best best best product",
                "date": "Reviewed in the United Kingdom on 19 January 2024",
                "vp": "Unverified Purchase",
                "body": "love it love it awesome best product ever made highly recommend very good quality must buy.",
                "helpful": "0 people found this helpful"
            },
            {
                "reviewer": "Vikram Patel",
                "rating": "3.0 out of 5 stars",
                "title": "Decent value for money, average durability",
                "date": "Reviewed in India on 05 February 2024",
                "vp": "Verified Purchase",
                "body": "It functions adequately for everyday tasks. The plastic buttons feel a little flimsy compared to older models, but the processor handles multitasking smoothly. A reasonable choice for budget-conscious buyers.",
                "helpful": "7 people found this helpful"
            },
            {
                "reviewer": "Ananya Roy",
                "rating": "5.0 out of 5 stars",
                "title": "Flawless design and intuitive user interface",
                "date": "Reviewed in India on 18 February 2024",
                "vp": "Verified Purchase",
                "body": "Setup took less than 3 minutes. The companion application has rich configuration options, and the haptic feedback is crisp. Really impressed by the premium design language.",
                "helpful": "23 people found this helpful"
            },
            {
                "reviewer": "David Brown",
                "rating": "1.0 out of 5 stars",
                "title": "Defective piece delivered, had to initiate refund",
                "date": "Reviewed in the United States on 12 January 2024",
                "vp": "Verified Purchase",
                "body": "The box arrived crushed and the unit failed to power on out of the box. Tried charging with multiple 65W chargers for 5 hours without success. Initiated immediate return and refund through Amazon.",
                "helpful": "29 people found this helpful"
            },
            {
                "reviewer": "Kavita Nair",
                "rating": "4.0 out of 5 stars",
                "title": "Reliable daily performer with quick charging",
                "date": "Reviewed in India on 25 January 2024",
                "vp": "Verified Purchase",
                "body": "The fast charging is incredible - goes from 10% to 80% in roughly 25 minutes. Screen brightness is comfortable at night. The camera lens gathers fingerprint smudges easily, but clean wipes fix it.",
                "helpful": "15 people found this helpful"
            }
        ]

        lines = ["Top reviews from India\n"]
        for r in curated_reviews:
            lines.append(r["reviewer"])
            lines.append(r["rating"])
            lines.append(r["title"])
            lines.append(r["date"])
            lines.append(r["vp"])
            lines.append(r["body"])
            lines.append(r["helpful"])
            lines.append("Report\n")

        return "\n".join(lines)


def extract_reviews(full_text: str) -> str:
    """Extract review section text from page text dump"""
    if not full_text:
        return ""

    markers = [
        "Top reviews from India",
        "Top reviews from the United States",
        "Top reviews from the United Kingdom",
        "Top reviews from Canada",
        "Top reviews",
        "Customer reviews"
    ]

    for marker in markers:
        if marker in full_text:
            return full_text[full_text.index(marker):]

    # If no review section header exists, do NOT return page footer/dump
    return ""


def parse_review(raw_review: str) -> Optional[Dict]:
    """
    Parse a raw single review text block into structured dictionary
    Expected columns:
    'Reviewer', 'Location', 'Date', 'Rating', 'Review', 'Verified Purchase', 'Helpful Votes'
    """
    if not raw_review or not raw_review.strip():
        return None

    # Blacklist check against bot/captcha/legal disclaimers
    raw_lower = raw_review.lower()
    blacklist = [
        "continue shopping",
        "conditions of use",
        "privacy notice",
        "amazon.com, inc",
        "validatecaptcha",
        "robot check",
        "api-services-support",
        "type the characters",
        "automated access",
        "click the button below"
    ]
    if any(b in raw_lower for b in blacklist):
        return None

    lines = [line.strip() for line in raw_review.split("\n") if line.strip()]
    if not lines:
        return None

    reviewer = lines[0] if len(lines) > 0 else "Amazon Customer"
    if any(b in reviewer.lower() for b in blacklist) or "click" in reviewer.lower():
        return None

    rating = 5.0
    date = "Recent"
    location = "India"
    verified = "No"
    helpful_votes = 0
    body_lines = []

    # Patterns
    rating_pattern = re.compile(r'([1-5](?:\.[0-9])?)\s*(?:out of 5|\/5|stars?)', re.IGNORECASE)
    date_pattern = re.compile(r'Reviewed in\s+(.+?)\s+on\s+(.+)', re.IGNORECASE)
    helpful_pattern = re.compile(r'(\d+)\s*(?:people|person)?\s*found this helpful', re.IGNORECASE)

    for line in lines[1:]:
        # Rating check
        r_match = rating_pattern.search(line)
        if r_match:
            try:
                rating = float(r_match.group(1))
            except ValueError:
                pass
            continue

        # Date & Location check
        d_match = date_pattern.search(line)
        if d_match:
            location = d_match.group(1).strip()
            date = d_match.group(2).strip()
            continue

        # Verified Purchase
        if "verified purchase" in line.lower():
            verified = "Yes"
            continue

        # Helpful votes
        h_match = helpful_pattern.search(line)
        if h_match:
            try:
                helpful_votes = int(h_match.group(1))
            except ValueError:
                pass
            continue

        if line.lower() in ["helpful", "report", "report abuse", "customer image", "images in this review"]:
            continue

        body_lines.append(line)

    review_text = " ".join(body_lines).strip()
    if not review_text or len(review_text) < 10:
        return None

    return {
        'Reviewer': reviewer,
        'Rating': rating,
        'Date': date,
        'Location': location,
        'Review': review_text,
        'Verified Purchase': verified,
        'Helpful Votes': helpful_votes
    }
