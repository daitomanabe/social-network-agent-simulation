"""Reality Diff: compare simulation predictions with real-world outcomes.

When the simulation runs ahead and reality "catches up", this module
computes how accurate the predictions were.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

from src.core.database import Database

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A recorded prediction from the simulation."""

    id: str = ""
    topic_id: str = ""
    predicted_for_date: str = ""  # The sim-date this prediction is about
    recorded_at_date: str = ""    # When the prediction was recorded (real date)
    recorded_at_step: int = 0

    # Predicted opinion distribution
    predicted_mean: float = 0.0
    predicted_std: float = 0.0
    predicted_distribution: dict[str, int] = field(default_factory=dict)

    # Predicted key events / trends
    predicted_trends: list[str] = field(default_factory=list)
    # e.g. ["Opinion polarization increasing", "Pro-regulation gaining momentum"]

    timeline_id: str = ""  # Which timeline made this prediction
    confidence: float = 0.5  # Self-assessed confidence (0-1)


@dataclass
class RealityOutcome:
    """A recorded real-world outcome for comparison."""

    id: str = ""
    topic_id: str = ""
    date: str = ""  # The real-world date

    # Observed reality
    actual_sentiment: float = 0.0  # Overall public sentiment (-1 to +1)
    actual_distribution: dict[str, int] = field(default_factory=dict)
    actual_events: list[str] = field(default_factory=list)
    # e.g. ["AI regulation bill passed with 60% approval"]

    source: str = ""  # Where this data came from


@dataclass
class DiffReport:
    """Comparison between a prediction and reality."""

    id: str = ""
    prediction_id: str = ""
    outcome_id: str = ""
    topic_id: str = ""
    date: str = ""

    # Accuracy metrics
    accuracy_score: float = 0.0        # 0-100%
    mean_opinion_error: float = 0.0    # Absolute error in mean opinion
    distribution_error: float = 0.0    # Earth-mover distance between distributions
    trend_accuracy: float = 0.0        # What % of predicted trends were correct

    # Narrative
    summary: str = ""                  # Human-readable diff summary
    correct_predictions: list[str] = field(default_factory=list)
    missed_predictions: list[str] = field(default_factory=list)
    surprises: list[str] = field(default_factory=list)  # Things we didn't predict

    # Meta
    days_ahead: int = 0  # How far ahead was the prediction?


class RealityDiffEngine:
    """Records predictions and compares them with real outcomes.

    Usage:
    1. During simulation: record_prediction() at each milestone
    2. When reality catches up: record_reality()
    3. Compute diff: compute_diff() generates accuracy report
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self.predictions: list[Prediction] = []
        self.outcomes: list[RealityOutcome] = []
        self.diff_reports: list[DiffReport] = []
        self._init_db_tables()

    def _init_db_tables(self) -> None:
        self.db.conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                predicted_for_date TEXT NOT NULL,
                recorded_at_date TEXT NOT NULL,
                recorded_at_step INTEGER,
                predicted_mean REAL,
                predicted_std REAL,
                predicted_distribution_json TEXT DEFAULT '{}',
                predicted_trends_json TEXT DEFAULT '[]',
                timeline_id TEXT,
                confidence REAL DEFAULT 0.5
            );

            CREATE TABLE IF NOT EXISTS reality_outcomes (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                date TEXT NOT NULL,
                actual_sentiment REAL,
                actual_distribution_json TEXT DEFAULT '{}',
                actual_events_json TEXT DEFAULT '[]',
                source TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS diff_reports (
                id TEXT PRIMARY KEY,
                prediction_id TEXT,
                outcome_id TEXT,
                topic_id TEXT,
                date TEXT,
                accuracy_score REAL,
                mean_opinion_error REAL,
                distribution_error REAL,
                trend_accuracy REAL,
                summary TEXT,
                correct_predictions_json TEXT DEFAULT '[]',
                missed_predictions_json TEXT DEFAULT '[]',
                surprises_json TEXT DEFAULT '[]',
                days_ahead INTEGER DEFAULT 0
            );
        """)
        self.db.conn.commit()

    def record_prediction(
        self,
        topic_id: str,
        predicted_for_date: str,
        recorded_at_date: str,
        recorded_at_step: int,
        mean_opinion: float,
        opinion_std: float,
        distribution: dict[str, int],
        trends: list[str] | None = None,
        timeline_id: str = "main",
        confidence: float = 0.5,
    ) -> Prediction:
        """Record a simulation prediction for a future date."""
        from uuid import uuid4
        pred = Prediction(
            id=uuid4().hex[:10],
            topic_id=topic_id,
            predicted_for_date=predicted_for_date,
            recorded_at_date=recorded_at_date,
            recorded_at_step=recorded_at_step,
            predicted_mean=mean_opinion,
            predicted_std=opinion_std,
            predicted_distribution=distribution,
            predicted_trends=trends or [],
            timeline_id=timeline_id,
            confidence=confidence,
        )

        self.db.conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pred.id, pred.topic_id, pred.predicted_for_date,
                pred.recorded_at_date, pred.recorded_at_step,
                pred.predicted_mean, pred.predicted_std,
                json.dumps(pred.predicted_distribution),
                json.dumps(pred.predicted_trends),
                pred.timeline_id, pred.confidence,
            ),
        )
        self.db.conn.commit()
        self.predictions.append(pred)

        logger.info(
            "Prediction recorded: %s for %s (mean=%.3f, step=%d)",
            pred.id, predicted_for_date, mean_opinion, recorded_at_step,
        )
        return pred

    def record_reality(
        self,
        topic_id: str,
        date: str,
        actual_sentiment: float,
        actual_distribution: dict[str, int] | None = None,
        actual_events: list[str] | None = None,
        source: str = "manual",
    ) -> RealityOutcome:
        """Record a real-world outcome for comparison."""
        from uuid import uuid4
        outcome = RealityOutcome(
            id=uuid4().hex[:10],
            topic_id=topic_id,
            date=date,
            actual_sentiment=actual_sentiment,
            actual_distribution=actual_distribution or {},
            actual_events=actual_events or [],
            source=source,
        )

        self.db.conn.execute(
            "INSERT INTO reality_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                outcome.id, outcome.topic_id, outcome.date,
                outcome.actual_sentiment,
                json.dumps(outcome.actual_distribution),
                json.dumps(outcome.actual_events),
                outcome.source,
            ),
        )
        self.db.conn.commit()
        self.outcomes.append(outcome)

        logger.info("Reality recorded: %s for %s (sentiment=%.2f)", outcome.id, date, actual_sentiment)
        return outcome

    def compute_diff(
        self,
        prediction: Prediction,
        outcome: RealityOutcome,
    ) -> DiffReport:
        """Compare a prediction with a real outcome."""
        from uuid import uuid4

        # Mean opinion error
        mean_error = abs(prediction.predicted_mean - outcome.actual_sentiment)

        # Distribution error (simplified Earth Mover's Distance)
        dist_error = _distribution_distance(
            prediction.predicted_distribution,
            outcome.actual_distribution,
        )

        # Trend accuracy (keyword matching)
        correct_trends = []
        missed_trends = []
        for trend in prediction.predicted_trends:
            # Check if any actual event matches this trend
            matched = any(
                _trend_matches(trend, event)
                for event in outcome.actual_events
            )
            if matched:
                correct_trends.append(trend)
            else:
                missed_trends.append(trend)

        trend_acc = (
            len(correct_trends) / len(prediction.predicted_trends)
            if prediction.predicted_trends else 0.5
        )

        # Surprises: actual events not predicted
        surprises = [
            e for e in outcome.actual_events
            if not any(_trend_matches(t, e) for t in prediction.predicted_trends)
        ]

        # Overall accuracy score (weighted)
        # Lower mean_error is better (max ~2.0 for [-1, +1] range)
        mean_acc = max(0, 1.0 - mean_error) * 100
        dist_acc = max(0, 1.0 - dist_error) * 100
        accuracy = mean_acc * 0.4 + dist_acc * 0.3 + trend_acc * 100 * 0.3

        # Days ahead
        try:
            pred_date = datetime.strptime(prediction.recorded_at_date, "%Y-%m-%d")
            outcome_date = datetime.strptime(outcome.date, "%Y-%m-%d")
            days_ahead = (outcome_date - pred_date).days
        except ValueError:
            days_ahead = 0

        # Generate summary
        summary = _generate_summary(
            prediction, outcome, mean_error, accuracy, days_ahead
        )

        report = DiffReport(
            id=uuid4().hex[:10],
            prediction_id=prediction.id,
            outcome_id=outcome.id,
            topic_id=prediction.topic_id,
            date=outcome.date,
            accuracy_score=round(accuracy, 1),
            mean_opinion_error=round(mean_error, 4),
            distribution_error=round(dist_error, 4),
            trend_accuracy=round(trend_acc, 4),
            summary=summary,
            correct_predictions=correct_trends,
            missed_predictions=missed_trends,
            surprises=surprises,
            days_ahead=days_ahead,
        )

        # Persist
        self.db.conn.execute(
            "INSERT INTO diff_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report.id, report.prediction_id, report.outcome_id,
                report.topic_id, report.date,
                report.accuracy_score, report.mean_opinion_error,
                report.distribution_error, report.trend_accuracy,
                report.summary,
                json.dumps(report.correct_predictions),
                json.dumps(report.missed_predictions),
                json.dumps(report.surprises),
                report.days_ahead,
            ),
        )
        self.db.conn.commit()
        self.diff_reports.append(report)

        return report

    def get_predictions_for_date(self, topic_id: str, date: str) -> list[Prediction]:
        """Find predictions that were made about a specific date."""
        return [
            p for p in self.predictions
            if p.topic_id == topic_id and p.predicted_for_date == date
        ]

    def get_accuracy_trend(self) -> list[dict]:
        """Get accuracy scores over time to see if predictions improve."""
        return [
            {
                "date": r.date,
                "accuracy": r.accuracy_score,
                "days_ahead": r.days_ahead,
                "topic": r.topic_id,
            }
            for r in sorted(self.diff_reports, key=lambda r: r.date)
        ]


def _distribution_distance(dist_a: dict, dist_b: dict) -> float:
    """Simplified distribution distance (normalized L1)."""
    all_keys = set(list(dist_a.keys()) + list(dist_b.keys()))
    if not all_keys:
        return 0.0

    total_a = max(sum(dist_a.values()), 1)
    total_b = max(sum(dist_b.values()), 1)

    total_diff = 0.0
    for key in all_keys:
        frac_a = dist_a.get(key, 0) / total_a
        frac_b = dist_b.get(key, 0) / total_b
        total_diff += abs(frac_a - frac_b)

    return total_diff / 2  # Normalize to [0, 1]


def _trend_matches(trend: str, event: str) -> bool:
    """Keyword-based trend matching with Japanese support.

    Japanese text doesn't split on spaces, so we use character n-gram
    overlap as a secondary matching method.
    """
    trend_lower = trend.lower()
    event_lower = event.lower()

    # Method 1: Word-level overlap (for space-separated text)
    trend_words = set(trend_lower.split())
    event_words = set(event_lower.split())
    if trend_words and event_words:
        overlap = trend_words & event_words
        if len(overlap) / len(trend_words) > 0.3:
            return True

    # Method 2: Substring matching (works for Japanese)
    # Check if significant substrings of the trend appear in the event
    # Use sliding window of 3-4 characters
    if len(trend_lower) >= 3 and len(event_lower) >= 3:
        # Count how many 3-char substrings of trend appear in event
        n = 3
        trend_ngrams = {trend_lower[i:i+n] for i in range(len(trend_lower) - n + 1)}
        matches = sum(1 for ng in trend_ngrams if ng in event_lower)
        if trend_ngrams and matches / len(trend_ngrams) > 0.3:
            return True

    # Method 3: Key Japanese keywords
    key_terms = _extract_key_terms(trend_lower)
    if key_terms:
        matched = sum(1 for term in key_terms if term in event_lower)
        if matched / len(key_terms) > 0.4:
            return True

    return False


def _extract_key_terms(text: str) -> list[str]:
    """Extract key terms from Japanese text (simple heuristic)."""
    # Common Japanese particles/connectors to skip
    skip = {"は", "が", "の", "を", "に", "で", "と", "も", "や", "か", "る", "い", "な", "た", "て", "し", "れ", "さ"}
    terms = []
    # Extract 2-4 character chunks that are likely meaningful
    for length in [4, 3, 2]:
        for i in range(len(text) - length + 1):
            chunk = text[i:i+length]
            if not any(c in skip for c in chunk) or length >= 3:
                terms.append(chunk)
    # Deduplicate and return top terms
    seen = set()
    unique = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:10]


def _generate_summary(
    prediction: Prediction,
    outcome: RealityOutcome,
    mean_error: float,
    accuracy: float,
    days_ahead: int,
) -> str:
    """Generate a human-readable diff summary."""
    lines = []
    lines.append(f"予測精度: {accuracy:.0f}% ({days_ahead}日先の予測)")

    if mean_error < 0.1:
        lines.append("世論の方向性: ほぼ正確に予測")
    elif mean_error < 0.3:
        lines.append(f"世論の方向性: 概ね正確（誤差: {mean_error:.2f}）")
    else:
        direction = "楽観的すぎ" if prediction.predicted_mean > outcome.actual_sentiment else "悲観的すぎ"
        lines.append(f"世論の方向性: {direction}（誤差: {mean_error:.2f}）")

    return "\n".join(lines)
