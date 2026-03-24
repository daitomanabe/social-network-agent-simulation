"""News ingestion from RSS feeds and manual input."""

from __future__ import annotations

import logging
from datetime import datetime

import httpx

from src.news.models import NewsItem

logger = logging.getLogger(__name__)


class NewsIngester:
    """Fetches news from various sources."""

    def __init__(self) -> None:
        self._http_client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def fetch_rss(self, url: str, max_items: int = 10) -> list[NewsItem]:
        """Fetch news items from an RSS feed URL."""
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed. Run: pip install feedparser")
            return []

        try:
            client = self._get_client()
            response = client.get(url)
            feed = feedparser.parse(response.text)
        except Exception as e:
            logger.error("Failed to fetch RSS from %s: %s", url, e)
            return []

        items = []
        for entry in feed.entries[:max_items]:
            item = NewsItem(
                title=entry.get("title", ""),
                summary=entry.get("summary", entry.get("description", "")),
                source=feed.feed.get("title", url),
                url=entry.get("link", ""),
                published=_parse_date(entry.get("published")),
                raw_content=entry.get("content", [{}])[0].get("value", "")
                if entry.get("content")
                else "",
            )
            items.append(item)

        logger.info("Fetched %d items from %s", len(items), url)
        return items

    def manual_inject(self, headline: str, summary: str, source: str = "manual") -> NewsItem:
        """Create a news item from manual input."""
        return NewsItem(
            title=headline,
            summary=summary,
            source=source,
            published=datetime.now(),
        )

    def close(self) -> None:
        if self._http_client:
            self._http_client.close()


class TopicExtractor:
    """Extracts simulation-ready topics from news items.

    In LLM mode, uses Claude to analyze the article. In simple mode,
    uses keyword matching.
    """

    # Simple keyword-based topic mapping
    TOPIC_KEYWORDS: dict[str, list[str]] = {
        "ai_regulation": ["AI", "人工知能", "規制", "regulation", "artificial intelligence", "LLM"],
        "climate_change": ["気候", "温暖化", "climate", "carbon", "CO2", "環境"],
        "labor_automation": ["雇用", "自動化", "労働", "automation", "employment", "ロボット"],
        "digital_privacy": ["プライバシー", "監視", "privacy", "surveillance", "データ保護"],
        "housing": ["住宅", "不動産", "housing", "家賃", "rent"],
    }

    def extract_simple(self, news_item: NewsItem, default_topic: str = "ai_regulation") -> dict:
        """Simple keyword-based topic extraction (no LLM needed)."""
        text = f"{news_item.title} {news_item.summary}".lower()

        # Find matching topic
        best_topic = default_topic
        best_score = 0
        for topic_id, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > best_score:
                best_score = score
                best_topic = topic_id

        # Simple sentiment: count positive/negative keywords
        positive = ["成功", "合意", "進展", "改善", "success", "progress", "improve"]
        negative = ["懸念", "反対", "問題", "危険", "concern", "oppose", "risk", "danger"]
        pos_count = sum(1 for w in positive if w in text)
        neg_count = sum(1 for w in negative if w in text)
        sentiment = (pos_count - neg_count) * 0.3
        sentiment = max(-1.0, min(1.0, sentiment))

        return {
            "topic_id": best_topic,
            "headline": news_item.title,
            "summary": news_item.summary[:200],
            "sentiment": sentiment,
            "key_claims": [],
        }


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        return None
