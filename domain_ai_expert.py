from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from sentence_transformers import SentenceTransformer

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List
from shared_resources import embedding_model

load_dotenv()

# Initialize OpenAI Model with explicit API key
model = OpenAIModel(
    'deepseek-chat',
    base_url='https://api.deepseek.com',
    api_key=os.getenv('DEEPSEEK_API_KEY')
)

if not os.getenv('DEEPSEEK_API_KEY'):
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    deepseek_client: AsyncOpenAI

system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework. You have access to all the documentation, including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this, and you don't answer other questions besides describing what you are able to do.
Don't ask the user before taking an action, just do it. 

When answering a user's question:
1. First, use the `retrieve_relevant_documentation` tool to find the most relevant documentation chunks.
2. Then use the `list_documentation_pages` tool to list all avialable urls pages in the dataset and identify relevant pages.
never use made up url webpages, only use the URLs provided by the `list_documentation_pages` tool.

3. if appropriate url is found from the list, Use the `get_page_content` tool to retrieve the full content of relvant documentation pages given by list_documentation_pages tool.

4. After retrieving the content, stop calling tools and provide the answer to the user.

Be concise and avoid unnecessary tool calls.
Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.

"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using stella_en_400M_v5."""
    try:
        embedding = embedding_model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1024  # stella_en_400M_v5 dimension is 1024

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and deepseek client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
        
        #print("Response from retrieve_relevant_documentation:", formatted_chunks)  # Inspect the response
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    This is should be called after using `list_documentation_pages` to get the URL.
    This is does not work for URLs that are not part of the documentation pages and found in the database.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"