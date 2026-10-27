"""
Product Comparison Analytics & Visual Dashboard Engine for Amazon Review Insight Pro
Enables side-by-side multi-product benchmarking across Sentiment, Authenticity, and Aspect Ratings.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class ProductComparator:
    """
    Multi-product analytical engine for benchmarking products side-by-side.
    """

    CORE_ASPECTS = ["Quality", "Performance", "Design", "Value", "Usability", "Service"]

    @classmethod
    def compute_product_summary(cls, name: str, df: pd.DataFrame, url: str = "") -> Dict[str, Any]:
        """Compute key analytical dimensions for a single product"""
        total = len(df)
        if total == 0:
            return {"name": name, "total": 0}

        avg_rating = float(df['Rating'].mean()) if 'Rating' in df else 0.0
        avg_polarity = float(df['Sentiment Polarity'].mean()) if 'Sentiment Polarity' in df else 0.0
        pos_pct = float(len(df[df['Sentiment'] == 'Positive']) / total * 100) if 'Sentiment' in df else 0.0
        neg_pct = float(len(df[df['Sentiment'] == 'Negative']) / total * 100) if 'Sentiment' in df else 0.0
        neutral_pct = float(len(df[df['Sentiment'] == 'Neutral']) / total * 100) if 'Sentiment' in df else 0.0

        genuine_count = len(df[df['Authenticity'] == 'Genuine']) if 'Authenticity' in df else 0
        genuine_pct = float(genuine_count / total * 100)
        avg_auth_score = float(df['Authenticity Score'].mean()) if 'Authenticity Score' in df else 50.0

        # Aspect Frequency & Sentiment Scores
        aspect_scores = {}
        for aspect in cls.CORE_ASPECTS:
            if 'Category' in df:
                aspect_mask = df['Category'].str.contains(aspect, case=False, na=False)
                aspect_subset = df[aspect_mask]
                if not aspect_subset.empty:
                    # Score is weighted by aspect rating and sentiment polarity (scaled 0 to 100)
                    mean_r = aspect_subset['Rating'].mean()
                    mean_pol = aspect_subset['Sentiment Polarity'].mean() if 'Sentiment Polarity' in aspect_subset else 0.0
                    score = (mean_r / 5.0 * 60) + ((mean_pol + 1.0) / 2.0 * 40)
                    aspect_scores[aspect] = round(score, 1)
                else:
                    aspect_scores[aspect] = round((avg_rating / 5.0 * 70), 1)
            else:
                aspect_scores[aspect] = 50.0

        return {
            "name": name,
            "url": url,
            "total_reviews": total,
            "avg_rating": round(avg_rating, 2),
            "avg_polarity": round(avg_polarity, 2),
            "positive_pct": round(pos_pct, 1),
            "negative_pct": round(neg_pct, 1),
            "neutral_pct": round(neutral_pct, 1),
            "genuine_pct": round(genuine_pct, 1),
            "avg_authenticity_score": round(avg_auth_score, 1),
            "aspect_scores": aspect_scores,
            "df": df
        }

    @classmethod
    def plot_aspect_radar(cls, products: List[Dict[str, Any]]) -> go.Figure:
        """Create a comparative Radar / Spider Chart across Core Product Aspects"""
        fig = go.Figure()

        palette = ['#6e8efb', '#00AA45', '#FFA500', '#FF4D4D', '#a777e3']

        for i, prod in enumerate(products):
            color = palette[i % len(palette)]
            scores = prod.get('aspect_scores', {})
            values = [scores.get(aspect, 50.0) for aspect in cls.CORE_ASPECTS]
            # Close the polygon loop
            values.append(values[0])
            theta = cls.CORE_ASPECTS + [cls.CORE_ASPECTS[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill='toself',
                name=prod['name'],
                line=dict(color=color, width=2.5),
                opacity=0.65
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                )
            ),
            showlegend=True,
            title="<b>Product Aspect Benchmark (Radar Analysis: 0-100 Score)</b>",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @classmethod
    def plot_metric_comparison_bars(cls, products: List[Dict[str, Any]]) -> go.Figure:
        """Grouped Bar Chart comparing Ratings, Sentiment Polarity (rescaled), and Authenticity %"""
        names = [p['name'] for p in products]
        ratings = [p['avg_rating'] for p in products]
        positives = [p['positive_pct'] for p in products]
        genuines = [p['genuine_pct'] for p in products]

        fig = go.Figure(data=[
            go.Bar(name='Average Star Rating (out of 5)', x=names, y=ratings, marker_color='#6e8efb'),
            go.Bar(name='Positive Sentiment %', x=names, y=[p / 20.0 for p in positives], # Scale to 5.0 for visual alignment
                   text=[f"{p:.1f}%" for p in positives], textposition='auto', marker_color='#00AA45'),
            go.Bar(name='Authentic Reviews %', x=names, y=[g / 20.0 for g in genuines], # Scale to 5.0
                   text=[f"{g:.1f}%" for g in genuines], textposition='auto', marker_color='#a777e3')
        ])

        fig.update_layout(
            barmode='group',
            title="<b>Side-by-Side Product Benchmarks (Ratings, Sentiment & Authenticity)</b>",
            yaxis=dict(title="Normalized Scale (0 to 5)"),
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @classmethod
    def generate_recommendation_verdict(cls, products: List[Dict[str, Any]]) -> Dict[str, str]:
        """Synthesize algorithmic recommendations across the compared products"""
        if not products:
            return {}

        best_rating = max(products, key=lambda p: p['avg_rating'])
        best_sentiment = max(products, key=lambda p: p['avg_polarity'])
        best_authenticity = max(products, key=lambda p: p['genuine_pct'])

        # Best value
        best_value = max(products, key=lambda p: p.get('aspect_scores', {}).get('Value', 0))

        return {
            "top_rated": f"🏆 **{best_rating['name']}** leads in Customer Satisfaction with an average {best_rating['avg_rating']}/5.0 stars.",
            "top_sentiment": f"❤️ **{best_sentiment['name']}** holds the strongest positive sentiment tone (polarity: {best_sentiment['avg_polarity']:.2f}).",
            "highest_trust": f"🛡️ **{best_authenticity['name']}** has the highest authenticity integrity with {best_authenticity['genuine_pct']}% genuine verified reviews.",
            "value_champion": f"💰 **{best_value['name']}** scored highest for consumer Value for Money."
        }
