from dataclasses import dataclass
from typing import Optional

@dataclass
class CrawlerConfig:
    # Can be either a local path like "/path/to/sitemap.xml" or remote URL like "https://example.com/sitemap.xml"
    sitemap_url: str = "/home/daba/Downloads/sitemap.xml"
    source_name: str = "mojoco_docs"  # For metadata
    chunk_size: int = 5000
    max_concurrent_crawls: int = 5
    test_mode_url_limit: int = 3
    generate_summaries: bool = False  # New flag to control title/summary generation
    ui_title: str = "AI Agentic RAG"
    ui_description: str = "Ask any question about the documentation, and I'll help you find the answers."
