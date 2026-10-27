"""
Sentiment Accuracy & Model Evaluation Suite for Amazon Review Insight Pro
Computes statistical ground-truth metrics, confusion matrix, precision/recall/F1, and discrepancy detection.
"""
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class SentimentEvaluator:
    """
    Evaluates NLP sentiment predictions against customer star ratings as proxy ground truth.
    
    Ground Truth Mapping:
    - 4.0 - 5.0 Stars -> Positive
    - 3.0 Stars       -> Neutral
    - 1.0 - 2.0 Stars -> Negative
    """

    CLASSES = ['Positive', 'Neutral', 'Negative']

    @staticmethod
    def map_rating_to_ground_truth(rating: float) -> str:
        """Map customer star rating to ground truth sentiment class"""
        try:
            r = float(rating)
        except (ValueError, TypeError):
            return "Neutral"

        if r >= 4.0:
            return "Positive"
        elif r == 3.0:
            return "Neutral"
        else:
            return "Negative"

    @classmethod
    def evaluate(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute full statistical evaluation report comparing predicted Sentiment with Rating.
        """
        if df.empty or 'Rating' not in df.columns or 'Sentiment' not in df.columns:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "class_metrics": {},
                "confusion_matrix": pd.DataFrame(),
                "discrepancies": pd.DataFrame(),
                "mae": 0.0,
                "correlation": 0.0
            }

        eval_df = df.copy()
        eval_df['Ground_Truth'] = eval_df['Rating'].apply(cls.map_rating_to_ground_truth)
        eval_df['Predicted'] = eval_df['Sentiment'].astype(str)

        total_samples = len(eval_df)
        matches = (eval_df['Ground_Truth'] == eval_df['Predicted']).sum()
        accuracy = (matches / total_samples * 100) if total_samples > 0 else 0.0

        # Construct 3x3 Confusion Matrix
        cm = pd.DataFrame(0, index=cls.CLASSES, columns=cls.CLASSES)
        for _, row in eval_df.iterrows():
            actual = row['Ground_Truth']
            pred = row['Predicted']
            if actual in cls.CLASSES and pred in cls.CLASSES:
                cm.loc[actual, pred] += 1

        # Calculate Per-Class Precision, Recall, F1
        class_metrics = {}
        f1_list = []
        weights = []

        for c in cls.CLASSES:
            tp = cm.loc[c, c]
            fp = cm[c].sum() - tp
            fn = cm.loc[c].sum() - tp
            support = cm.loc[c].sum()

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            class_metrics[c] = {
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1-Score": round(f1, 3),
                "Support": int(support),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn)
            }
            f1_list.append(f1)
            weights.append(support)

        macro_f1 = np.mean(f1_list) if f1_list else 0.0
        weighted_f1 = (np.average(f1_list, weights=weights)) if (weights and sum(weights) > 0) else 0.0

        # Mean Absolute Error between normalized rating and polarity
        mae = 0.0
        correlation = 0.0
        if 'Sentiment Polarity' in eval_df.columns:
            try:
                # Normalize Rating (1-5) to 0.0-1.0
                norm_rating = (eval_df['Rating'].astype(float) - 1.0) / 4.0
                # Normalize Polarity (-1.0 to 1.0) to 0.0-1.0
                norm_polarity = (eval_df['Sentiment Polarity'].astype(float) + 1.0) / 2.0
                mae = np.mean(np.abs(norm_rating - norm_polarity))
                if norm_rating.std() > 0 and norm_polarity.std() > 0:
                    correlation = np.corrcoef(norm_rating, norm_polarity)[0, 1]
            except Exception:
                pass

        # Detect High Discrepancies (Sarcasm, Polarity Inversion, or Mixed Feedback)
        high_discrepancies = eval_df[
            ((eval_df['Ground_Truth'] == 'Positive') & (eval_df['Predicted'] == 'Negative')) |
            ((eval_df['Ground_Truth'] == 'Negative') & (eval_df['Predicted'] == 'Positive'))
        ].copy()

        if not high_discrepancies.empty:
            def label_discrepancy(row):
                if row['Ground_Truth'] == 'Positive' and row['Predicted'] == 'Negative':
                    return "5-Star Rating but Critical/Negative Text (High Sarcasm or Inversion Risk)"
                else:
                    return "1/2-Star Rating but Positive Text (Possible Promotional or Fake Anomaly)"
            high_discrepancies['Discrepancy_Type'] = high_discrepancies.apply(label_discrepancy, axis=1)

        return {
            "total_evaluated": total_samples,
            "accuracy": round(accuracy, 2),
            "macro_f1": round(macro_f1, 3),
            "weighted_f1": round(weighted_f1, 3),
            "mae": round(float(mae), 3),
            "correlation": round(float(correlation), 3),
            "class_metrics": class_metrics,
            "confusion_matrix": cm,
            "discrepancies": high_discrepancies,
            "evaluated_df": eval_df
        }

    @classmethod
    def plot_confusion_matrix(cls, cm: pd.DataFrame) -> go.Figure:
        """Create an annotated Plotly Heatmap of the Sentiment Confusion Matrix"""
        z_vals = cm.values
        x_labels = [f"Pred: {col}" for col in cm.columns]
        y_labels = [f"Actual: {idx}" for idx in cm.index]

        # Annotations text with percentages
        total = cm.values.sum()
        annotations = []
        for i, row in enumerate(cm.index):
            for j, col in enumerate(cm.columns):
                val = cm.loc[row, col]
                pct = (val / total * 100) if total > 0 else 0
                annotations.append(
                    dict(
                        x=x_labels[j],
                        y=y_labels[i],
                        text=f"<b>{val}</b><br><span style='font-size:10px;'>({pct:.1f}%)</span>",
                        showarrow=False,
                        font=dict(color="white" if val > (total * 0.25) else "#121b2e")
                    )
                )

        fig = go.Figure(
            data=go.Heatmap(
                z=z_vals,
                x=x_labels,
                y=y_labels,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Review Count")
            )
        )
        fig.update_layout(
            title="<b>Sentiment Confusion Matrix (Rating Ground Truth vs NLP Prediction)</b>",
            xaxis_title="Predicted Sentiment",
            yaxis_title="Actual Sentiment (Derived from Star Rating)",
            annotations=annotations,
            height=420,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @classmethod
    def plot_class_performance(cls, class_metrics: Dict[str, Dict]) -> go.Figure:
        """Create comparative bar chart for Precision, Recall, and F1 across sentiment classes"""
        classes = list(class_metrics.keys())
        precisions = [class_metrics[c]['Precision'] for c in classes]
        recalls = [class_metrics[c]['Recall'] for c in classes]
        f1s = [class_metrics[c]['F1-Score'] for c in classes]

        fig = go.Figure(data=[
            go.Bar(name='Precision', x=classes, y=precisions, marker_color='#6e8efb'),
            go.Bar(name='Recall', x=classes, y=recalls, marker_color='#a777e3'),
            go.Bar(name='F1-Score', x=classes, y=f1s, marker_color='#00AA45')
        ])
        fig.update_layout(
            barmode='group',
            title="<b>Precision, Recall & F1-Score by Sentiment Class</b>",
            yaxis=dict(title="Score (0.0 to 1.0)", range=[0, 1.05]),
            xaxis=dict(title="Sentiment Category"),
            height=380,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @classmethod
    def plot_rating_vs_polarity(cls, df: pd.DataFrame) -> go.Figure:
        """Scatter plot with jitter correlating Star Rating and Sentiment Polarity"""
        plot_df = df.copy()
        # Add slight jitter to ratings for visual clarity
        plot_df['Jittered_Rating'] = plot_df['Rating'] + np.random.uniform(-0.12, 0.12, size=len(plot_df))
        
        fig = px.scatter(
            plot_df,
            x='Jittered_Rating',
            y='Sentiment Polarity',
            color='Sentiment',
            color_discrete_map={'Positive': '#00AA45', 'Negative': '#FF4D4D', 'Neutral': '#FFA500'},
            hover_data=['Reviewer', 'Rating', 'Sentiment Polarity', 'Review'],
            title="<b>Customer Star Rating vs NLP Sentiment Polarity Correlation</b>"
        )
        fig.update_layout(
            xaxis=dict(title="Customer Star Rating (1 to 5)", tickvals=[1, 2, 3, 4, 5]),
            yaxis=dict(title="Sentiment Polarity (-1.0 to +1.0)"),
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig
