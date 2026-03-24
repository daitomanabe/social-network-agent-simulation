"""Automated news fetching and injection scheduler.

Periodically polls RSS feeds for new articles, extracts topics,
and injects them into the running simulation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from src.news.ingestion import NewsIngester, TopicExtractor
from src.news.models import NewsItem

logger = logging.getLogger(__name__)


@dataclass
class FeedSource:
    """An RSS feed source configuration."""

    name: str
    url: str
    poll_interval_seconds: int = 600  # 10 minutes
    max_items_per_poll: int = 5
    enabled: bool = True
    last_polled: float = 0.0
    seen_urls: set[str] = field(default_factory=set)


# Curated list of Japanese and international news feeds
DEFAULT_FEEDS = [
    FeedSource(name="NHK 主要", url="https://www3.nhk.or.jp/rss/news/cat0.xml"),
    FeedSource(name="NHK 科学", url="https://www3.nhk.or.jp/rss/news/cat7.xml"),
    FeedSource(name="Reuters Top", url="https://feeds.reuters.com/reuters/topNews"),
    FeedSource(name="Reuters Tech", url="https://feeds.reuters.com/reuters/technologyNews"),
]


class NewsScheduler:
    """Manages automated news polling and injection.

    Can run in:
    - Sync mode: call poll_all() periodically from the simulation loop
    - Async mode: run as a background task with start_background()
    """

    def __init__(
        self,
        feeds: list[FeedSource] | None = None,
        auto_inject: bool = True,
    ) -> None:
        self.feeds = feeds or list(DEFAULT_FEEDS)
        self.ingester = NewsIngester()
        self.extractor = TopicExtractor()
        self.auto_inject = auto_inject

        self._pending_news: list[dict] = []  # Processed news waiting for injection
        self._history: list[dict] = []  # All processed news
        self._running = False
        self._inject_callback = None  # Function to call to inject news

    def set_inject_callback(self, callback) -> None:
        """Set the function to call when news should be injected.

        Callback signature: callback(headline, summary, sentiment, topic_id)
        """
        self._inject_callback = callback

    def add_feed(self, name: str, url: str, interval: int = 600) -> None:
        """Add a new RSS feed source."""
        self.feeds.append(FeedSource(name=name, url=url, poll_interval_seconds=interval))

    def poll_all(self) -> list[dict]:
        """Poll all feeds that are due for checking.

        Returns list of new processed news items.
        """
        now = time.monotonic()
        new_items = []

        for feed in self.feeds:
            if not feed.enabled:
                continue
            if now - feed.last_polled < feed.poll_interval_seconds:
                continue

            try:
                items = self._poll_feed(feed)
                new_items.extend(items)
                feed.last_polled = now
            except Exception as e:
                logger.error("Failed to poll %s: %s", feed.name, e)

        # Auto-inject if callback is set
        if self.auto_inject and self._inject_callback and new_items:
            for item in new_items:
                try:
                    self._inject_callback(
                        item["headline"],
                        item["summary"],
                        item["sentiment"],
                        item.get("topic_id"),
                    )
                    item["injected"] = True
                    logger.info("Auto-injected: %s", item["headline"][:50])
                except Exception as e:
                    logger.error("Failed to inject news: %s", e)
                    item["injected"] = False

        self._history.extend(new_items)
        return new_items

    def _poll_feed(self, feed: FeedSource) -> list[dict]:
        """Poll a single feed and process new items."""
        raw_items = self.ingester.fetch_rss(feed.url, max_items=feed.max_items_per_poll)

        new_items = []
        for item in raw_items:
            if item.url in feed.seen_urls:
                continue
            if item.url:
                feed.seen_urls.add(item.url)

            # Extract topic
            extracted = self.extractor.extract_simple(item)

            processed = {
                "headline": item.title,
                "summary": extracted["summary"],
                "topic_id": extracted["topic_id"],
                "sentiment": extracted["sentiment"],
                "source": feed.name,
                "url": item.url,
                "published": item.published.isoformat() if item.published else "",
                "fetched_at": datetime.now().isoformat(),
                "injected": False,
            }
            new_items.append(processed)

        if new_items:
            logger.info("Fetched %d new items from %s", len(new_items), feed.name)

        return new_items

    def poll_manual(self, headline: str, summary: str = "", sentiment: float = 0.0) -> dict:
        """Manually add a news item to the queue."""
        item = {
            "headline": headline,
            "summary": summary or headline,
            "topic_id": None,
            "sentiment": sentiment,
            "source": "manual",
            "url": "",
            "published": "",
            "fetched_at": datetime.now().isoformat(),
            "injected": False,
        }

        # Try to detect topic
        news_item = self.ingester.manual_inject(headline, summary)
        extracted = self.extractor.extract_simple(news_item)
        item["topic_id"] = extracted["topic_id"]
        if sentiment == 0.0:
            item["sentiment"] = extracted["sentiment"]

        self._pending_news.append(item)
        self._history.append(item)

        if self.auto_inject and self._inject_callback:
            self._inject_callback(
                item["headline"], item["summary"],
                item["sentiment"], item["topic_id"],
            )
            item["injected"] = True

        return item

    async def start_background(self, interval_seconds: int = 60) -> None:
        """Run polling as a background async task."""
        self._running = True
        logger.info("News scheduler started (interval=%ds)", interval_seconds)

        while self._running:
            try:
                new_items = self.poll_all()
                if new_items:
                    logger.info("Background poll found %d new items", len(new_items))
            except Exception as e:
                logger.error("Background poll error: %s", e)

            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        """Stop the background polling."""
        self._running = False

    @property
    def pending_count(self) -> int:
        return len(self._pending_news)

    @property
    def history(self) -> list[dict]:
        return self._history

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "feeds": len(self.feeds),
            "enabled_feeds": sum(1 for f in self.feeds if f.enabled),
            "total_fetched": len(self._history),
            "total_injected": sum(1 for h in self._history if h.get("injected")),
            "pending": self.pending_count,
        }
