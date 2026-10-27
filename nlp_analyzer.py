"""
High-performance NLP analysis functionality for Amazon Review Insight Pro
Optimized for low-latency batch analysis and zero-redundancy single-pass inference.
"""
import logging
import sys
from typing import Dict, List, Tuple, Optional
from collections import Counter
from functools import lru_cache
from textblob import TextBlob

logger = logging.getLogger(__name__)

class MiniToken:
    """Lightweight token wrapper mimicking spaCy token interface"""
    def __init__(self, text, pos="NOUN", tag="", dep=""):
        self.text = text
        self.pos_ = pos
        self.tag_ = tag
        self.dep_ = dep
        self.like_num = text.replace('.', '', 1).isdigit()

class MiniDoc:
    """Lightweight document wrapper for pure-python TextBlob fallback"""
    def __init__(self, text):
        blob = TextBlob(text)
        self.tokens = []
        pos_map = {
            'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB',
            'CD': 'NUM', 'RB': 'ADV', 'RBR': 'ADV', 'PRP': 'PRON', 'PRP$': 'PRON'
        }
        try:
            for word, tag in blob.tags:
                pos = pos_map.get(tag, pos_map.get(tag[:2], 'NOUN'))
                self.tokens.append(MiniToken(word, pos=pos, tag=tag))
        except Exception:
            for word in text.split():
                self.tokens.append(MiniToken(word, pos='NOUN'))

        self.ents = []
        try:
            self.noun_chunks = [MiniToken(chunk) for chunk in blob.noun_phrases]
            self.sents = [MiniToken(str(s)) for s in blob.sentences]
        except Exception:
            self.noun_chunks = []
            self.sents = [MiniToken(text)]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

class NLPAnalyzer:
    """Enhanced, high-throughput NLP analyzer with single-pass document caching and cloud fallbacks"""
    
    def __init__(self, config: dict):
        self.config = config
        self.nlp = self._load_spacy_model()
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "is", "was"}

    def _load_spacy_model(self):
        """Load spaCy model with fast fallback if spaCy is unavailable"""
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.info(f"spaCy not available or not installed ({e}); using high-speed TextBlob fallback.")
            return None

    def parse_doc(self, text: str):
        """Unified document parser handling both spaCy and pure-python MiniDoc fallback"""
        if self.nlp is not None:
            try:
                return self.nlp(text)
            except Exception:
                pass
        return MiniDoc(text)

    def analyze_sentiment(self, review_text: str, doc=None) -> Tuple[str, str, float, float]:
        """Fast sentiment analysis using TextBlob and spaCy aspects"""
        blob = TextBlob(review_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        pos_thresh = self.config.get('nlp', {}).get('sentiment_threshold_positive', 0.2)
        neg_thresh = self.config.get('nlp', {}).get('sentiment_threshold_negative', -0.2)

        if polarity > pos_thresh:
            sentiment = "Positive"
            reason = f"Strong positive sentiment (polarity: {polarity:.2f})"
        elif polarity < neg_thresh:
            sentiment = "Negative"
            reason = f"Strong negative sentiment (polarity: {polarity:.2f})"
        else:
            sentiment = "Neutral"
            reason = f"Neutral sentiment (polarity: {polarity:.2f})"
        
        if doc is None:
            doc = self.parse_doc(review_text)

        # Extract top aspects without re-parsing
        aspects = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN") and token.text.lower() not in self.stop_words]
        if aspects:
            top_aspects = [asp for asp, _ in Counter(aspects).most_common(3)]
            reason += f". Main aspects: {', '.join(top_aspects)}"
        
        return sentiment, reason, polarity, subjectivity

    def categorize_review_content(self, review_text: str, doc=None) -> str:
        """Categorize review into product aspects using keywords and spaCy tokens"""
        if doc is None:
            doc = self.parse_doc(review_text)

        categories = {
            "Quality": ("quality", "material", "build", "durable", "sturdy", "flimsy", "cheap"),
            "Performance": ("performance", "speed", "fast", "slow", "efficient", "powerful"),
            "Design": ("design", "look", "appearance", "style", "color", "shape", "size"),
            "Value": ("price", "cost", "worth", "value", "expensive", "affordable"),
            "Features": ("feature", "function", "setting", "mode", "option", "specification"),
            "Usability": ("easy", "difficult", "simple", "complex", "user-friendly", "interface"),
            "Service": ("delivery", "service", "support", "warranty", "return", "refund")
        }

        detected = set()
        text_lower = review_text.lower()

        # Token match
        for token in doc:
            t_lower = token.text.lower()
            for cat, keywords in categories.items():
                if t_lower in keywords:
                    detected.add(cat)

        # Fast substring fallback if tokens yielded nothing
        if not detected:
            for cat, keywords in categories.items():
                if any(kw in text_lower for kw in keywords):
                    detected.add(cat)

        return ", ".join(detected) if detected else "General"

    def analyze_linguistic_patterns(self, review_text: str, doc=None) -> Dict[str, int]:
        """Analyze linguistic patterns for authenticity detection using existing doc"""
        if doc is None:
            doc = self.parse_doc(review_text)

        features = {
            "excessive_adjectives": 0,
            "first_person_pronouns": 0,
            "passive_voice": 0,
            "specific_details": 0,
            "emotional_language": 0,
            "generic_phrases": 0
        }

        generic_phrases = self.config.get('nlp', {}).get('generic_phrases', [
            "great product", "highly recommend", "value for money", 
            "must buy", "good quality", "as described", "happy with purchase"
        ])
        emotional_words = {"love", "hate", "awesome", "terrible", "disappointed", "thrilled", "horrible", "best"}
        pronouns = {"i", "me", "my", "mine", "we", "our"}

        for token in doc:
            pos = token.pos_
            t_lower = token.text.lower()
            if pos == "ADJ":
                features["excessive_adjectives"] += 1
            if t_lower in pronouns:
                features["first_person_pronouns"] += 1
            if token.dep_ == "auxpass":
                features["passive_voice"] += 1
            if token.like_num:
                features["specific_details"] += 1
            if t_lower in emotional_words:
                features["emotional_language"] += 1

        text_lower = review_text.lower()
        features["generic_phrases"] = sum(phrase in text_lower for phrase in generic_phrases)
        return features

    def check_authenticity(self, review: dict, doc=None) -> Tuple[str, str, float]:
        """Enhanced authenticity check using NLP patterns with doc reuse"""
        text = str(review.get('Review', ''))
        features = self.analyze_linguistic_patterns(text, doc=doc)

        score = 0
        score += min(features["first_person_pronouns"], 5) * 2
        score += min(features["specific_details"], 3) * 5
        score -= min(features["excessive_adjectives"], 10)
        score -= min(features["generic_phrases"], 5) * 3
        score -= min(features["emotional_language"], 5) * 2

        normalized_score = max(0, min(100, 50 + score))

        high_thresh = self.config.get('nlp', {}).get('authenticity_threshold_high', 70)
        med_thresh = self.config.get('nlp', {}).get('authenticity_threshold_medium', 40)

        if normalized_score >= high_thresh:
            authenticity = "Genuine"
            reason = f"High authenticity score ({normalized_score}/100). Shows personal experience and specific details."
        elif normalized_score >= med_thresh:
            authenticity = "Likely Genuine"
            reason = f"Moderate authenticity score ({normalized_score}/100). Some indicators of personal experience."
        else:
            authenticity = "Potentially Fake"
            reason = f"Low authenticity score ({normalized_score}/100). Contains patterns common in fake reviews."

        return authenticity, reason, normalized_score

    def process_review_unified(self, review: dict) -> dict:
        """
        Processes a single review in a single unified spaCy pass.
        Reuses the parsed Document across sentiment, categorization, and authenticity.
        """
        raw_text = str(review.get('Cleaned_Review', '') or review.get('Review', ''))
        doc = self.parse_doc(raw_text)

        sentiment, s_reason, polarity, subjectivity = self.analyze_sentiment(raw_text, doc=doc)
        category = self.categorize_review_content(raw_text, doc=doc)
        authenticity, a_reason, a_score = self.check_authenticity(review, doc=doc)

        return {
            'Sentiment': sentiment,
            'Sentiment Analysis': s_reason,
            'Sentiment Polarity': polarity,
            'Sentiment Subjectivity': subjectivity,
            'Category': category,
            'Authenticity': authenticity,
            'Authenticity Reason': a_reason,
            'Authenticity Score': a_score
        }

    def get_linguistic_features(self, review_text: str) -> Dict:
        """Get detailed linguistic features for review inspector"""
        doc = self.parse_doc(review_text)
        return {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_tags': [(token.text, token.pos_, token.tag_) for token in doc],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'sentences': [sent.text for sent in doc.sents],
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents))
        }
