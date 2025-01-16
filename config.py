from dataclasses import dataclass
from typing import Optional

@dataclass
class CrawlerConfig:
    sitemap_url: str = "https://ai.pydantic.dev/sitemap.xml"
    source_name: str = "pydantic_ai_docs"  # For metadata
    chunk_size: int = 5000
    max_concurrent_crawls: int = 5
    test_mode_url_limit: int = 3
    generate_summaries: bool = False  # New flag to control title/summary generation
    ui_title: str = "AI Agentic RAG"
    ui_description: str = "Ask any question about the documentation, and I'll help you find the answers."
