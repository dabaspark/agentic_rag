import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

from sentence_transformers import SentenceTransformer
import sys
from shared_resources import embedding_model
from config import CrawlerConfig

load_dotenv()

# Initialize deepseek and Supabase clients

# Initialize DeepSeek client
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = CrawlerConfig.chunk_size) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using DeepSeek."""
    if not CrawlerConfig.generate_summaries:
        return {"title": "Untitled", "summary": "No summary generated"}
        
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Extremely Short bullet points of the contents description of the chunks  (6 words maximum per different contents)
    Keep both title and summary concise but informative."""
    
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using stella_en_400M_v5."""
    try:
        embedding = embedding_model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1024  # stella_en_400M_v5 dimension is 1024

async def process_chunk(chunk: str, chunk_number: int, url: str, source_name: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": source_name,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str, source_name: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url, source_name) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], config: CrawlerConfig):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(config.max_concurrent_crawls)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown, config.source_name)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_docs_urls(config: CrawlerConfig, testing: bool = False) -> List[str]:
    """Get URLs from docs sitemap.
    
    Args:
        testing (bool): If True, returns only first n URLs for testing purposes
    """
    try:
        # Handle both local and remote sitemaps
        if config.sitemap_url.startswith(('http://', 'https://')):
            # Remote sitemap
            response = requests.get(config.sitemap_url)
            response.raise_for_status()
            sitemap_content = response.content
        else:
            # Local sitemap - remove file:// prefix if present
            local_path = config.sitemap_url.replace('file://', '')
            with open(local_path, 'rb') as f:
                sitemap_content = f.read()
        
        # Parse the XML
        root = ElementTree.fromstring(sitemap_content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        # If in testing mode, return only first n URLs
        if testing:
            print(f"Running in testing mode - using only first {config.test_mode_url_limit} URLs")
            return urls[:config.test_mode_url_limit]
            
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Get command line arguments
    testing_mode = "--test" in sys.argv
    
    # Get URLs from docs
    urls = get_docs_urls(CrawlerConfig, testing=testing_mode)
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls, CrawlerConfig)

if __name__ == "__main__":
    asyncio.run(main())
