# Agentic RAG

I built this because I was tired of:
1. LLMs being clueless about newer APIs when programming
2. Basic RAG setups choking on technical stuff

## Why This solution

- **Works with Any Docs**: Point it at any documentation site and go
- **Completely Free Embeddings**: Using stella_en_400M_v5 - it's a beast at retrieval and totally working locally
- **Dirt Cheap LLM**: DeepSeek's API costs next to nothing
- **Free Database**: Supabase's free tier has everything we need

## Getting Started

1. Clone and set up:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Your keys in `.env.example`: (change the name from .env.example to .env as well)
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

3. Quick config in `config.py`:
```python
class CrawlerConfig:
    source_name: str = "your_docs_name"
    sitemap_url: str = "your_sitemap_url"
    # ...other settings if you want to tweak things see `config.py`
```

## Working with Sitemaps

You have several options to get a sitemap:

1. Direct URL: Most sites have a sitemap at `https://example.com/sitemap.xml`

2. Generate locally using Python script or use https://www.xml-sitemaps.com/ to generate it for you.

## Quick Start

1. Set up the database (just copy-paste the SQL from `site_pages.sql` into Supabase)

2. Grab the docs:
```bash
python docs_crawler.py  # add --test to try with just a few pages first
```

3. Fire up the chat:
```bash
streamlit run streamlit_ui.py
```

## How I Use This

1. Find some docs I need to work with
2. Let it crawl (it's pretty quick)
3. Start asking questions about the API/framework/whatever

## What's Inside

Just a few key files:
- `docs_crawler.py`: Grabs and processes the docs
- `domain_ai_expert.py`: The brains of the operation
- `streamlit_ui.py`: Simple chat interface
- `config.py`: Your settings live here
- `shared_resources.py`: Handles the embedding model


Feel free to use this for your own documentation needs or contribute improvements!
