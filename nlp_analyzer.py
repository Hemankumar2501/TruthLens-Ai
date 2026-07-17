"""
NLP analysis functionality for Amazon Review Insight Pro
"""
import logging
from typing import Dict, List, Tuple
from collections import Counter
from textblob import TextBlob
import spacy
from spacy import displacy
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class NLPAnalyzer:
    """Enhanced NLP analyzer for review sentiment and authenticity detection"""
    
    def __init__(self, config: dict):
        self.config = config
        self.nlp = self._load_spacy_model()
        self.stop_words = set(stopwords.words('english'))
        
        # Download NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK resources: {e}")
    
    def _load_spacy_model(self):
        """Load spaCy model with fallback"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
    
    def analyze_sentiment(self, review_text: str) -> Tuple[str, str, float, float]:
        """Advanced sentiment analysis using TextBlob and spaCy"""
        # TextBlob analysis
        blob = TextBlob(review_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > self.config['nlp']['sentiment_threshold_positive']:
            sentiment = "Positive"
            reason = f"Strong positive sentiment (polarity: {polarity:.2f})"
        elif polarity < self.config['nlp']['sentiment_threshold_negative']:
            sentiment = "Negative"
            reason = f"Strong negative sentiment (polarity: {polarity:.2f})"
        else:
            sentiment = "Neutral"
            reason = f"Neutral sentiment (polarity: {polarity:.2f})"
        
        # Additional insights from spaCy
        doc = self.nlp(review_text)
        
        # Extract key aspects being discussed
        aspects = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in self.stop_words:
                aspects.append(token.text)
        
        # Count most frequent aspects
        aspect_counts = Counter(aspects)
        top_aspects = [aspect for aspect, count in aspect_counts.most_common(3)]
        
        if top_aspects:
            reason += f". Main aspects: {', '.join(top_aspects)}"
        
        return sentiment, reason, polarity, subjectivity
    
    def categorize_review_content(self, review_text: str) -> str:
        """Categorize review into product aspects using NLP"""
        doc = self.nlp(review_text.lower())
        
        categories = {
            "Quality": ["quality", "material", "build", "durable", "sturdy", "flimsy", "cheap"],
            "Performance": ["performance", "speed", "fast", "slow", "efficient", "powerful"],
            "Design": ["design", "look", "appearance", "style", "color", "shape", "size"],
            "Value": ["price", "cost", "worth", "value", "expensive", "affordable"],
            "Features": ["feature", "function", "setting", "mode", "option", "specification"],
            "Usability": ["easy", "difficult", "simple", "complex", "user-friendly", "interface"],
            "Service": ["delivery", "service", "support", "warranty", "return", "refund"]
        }
        
        detected_categories = set()
        
        # Check for category keywords
        for token in doc:
            for category, keywords in categories.items():
                if token.text in keywords:
                    detected_categories.add(category)
        
        # If no categories detected, try pattern matching
        if not detected_categories:
            for category, keywords in categories.items():
                if any(keyword in review_text for keyword in keywords):
                    detected_categories.add(category)
        
        # If still no categories, use noun chunks
        if not detected_categories:
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            for chunk in noun_chunks:
                for category, keywords in categories.items():
                    if any(keyword in chunk for keyword in keywords):
                        detected_categories.add(category)
        
        return ", ".join(detected_categories) if detected_categories else "General"
    
    def analyze_linguistic_patterns(self, review_text: str) -> Dict[str, int]:
        """Analyze linguistic patterns for authenticity detection"""
        doc = self.nlp(review_text)
        
        # Features to examine
        features = {
            "excessive_adjectives": 0,
            "first_person_pronouns": 0,
            "passive_voice": 0,
            "specific_details": 0,
            "emotional_language": 0,
            "generic_phrases": 0
        }
        
        # Common generic phrases from config
        generic_phrases = self.config.get('nlp', {}).get('generic_phrases', [
            "great product", "highly recommend", "value for money", 
            "must buy", "good quality", "as described", "happy with purchase"
        ])
        
        # Count features
        for token in doc:
            # Excessive adjectives
            if token.pos_ == "ADJ":
                features["excessive_adjectives"] += 1
            
            # First person pronouns
            if token.text.lower() in ["i", "me", "my", "mine", "we", "our"]:
                features["first_person_pronouns"] += 1
            
            # Passive voice detection (simplified)
            if token.dep_ == "auxpass":
                features["passive_voice"] += 1
        
        # Check for specific details (numbers, measurements, etc.)
        if any(token.like_num for token in doc):
            features["specific_details"] += 1
        
        # Check for emotional language
        emotional_words = ["love", "hate", "awesome", "terrible", "disappointed", "thrilled"]
        if any(token.text.lower() in emotional_words for token in doc):
            features["emotional_language"] += 1
        
        # Check for generic phrases
        features["generic_phrases"] = sum(phrase in review_text.lower() for phrase in generic_phrases)
        
        return features
    
    def check_authenticity(self, review: dict) -> Tuple[str, str, float]:
        """Enhanced authenticity check using NLP patterns"""
        text = review['Review'].lower()
        features = self.analyze_linguistic_patterns(text)
        
        # Calculate authenticity score
        score = 0
        
        # Positive indicators
        score += min(features["first_person_pronouns"], 5) * 2  # Max 10 points
        score += min(features["specific_details"], 3) * 5  # Max 15 points
        
        # Negative indicators
        score -= min(features["excessive_adjectives"], 10)  # Max -10 points
        score -= min(features["generic_phrases"], 5) * 3  # Max -15 points
        score -= min(features["emotional_language"], 5) * 2  # Max -10 points
        
        # Normalize score to 0-100 range
        normalized_score = max(0, min(100, 50 + score))
        
        # Determine authenticity based on config thresholds
        high_threshold = self.config['nlp']['authenticity_threshold_high']
        medium_threshold = self.config['nlp']['authenticity_threshold_medium']
        
        if normalized_score >= high_threshold:
            authenticity = "Genuine"
            reason = f"High authenticity score ({normalized_score}/100). Shows personal experience and specific details."
        elif normalized_score >= medium_threshold:
            authenticity = "Likely Genuine"
            reason = f"Moderate authenticity score ({normalized_score}/100). Some indicators of personal experience."
        else:
            authenticity = "Potentially Fake"
            reason = f"Low authenticity score ({normalized_score}/100). Contains patterns common in fake reviews."
        
        return authenticity, reason, normalized_score
    
    def get_linguistic_features(self, review_text: str) -> Dict:
        """Get detailed linguistic features for a review"""
        doc = self.nlp(review_text)
        
        return {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_tags': [(token.text, token.pos_, token.tag_) for token in doc],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'sentences': [sent.text for sent in doc.sents],
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents))
        }
