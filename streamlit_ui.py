from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from domain_ai_expert import documentation_expert, PydanticAIDeps
from shared_resources import embedding_model
from config import CrawlerConfig

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Validate environment variables
if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"):
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file")

supabase: Client = Client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_SERVICE_KEY", "")
)

# Initialize deepseek clients for llm
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_and_display(user_input: str):
    """Run the agent and display the complete response at once."""
    config = CrawlerConfig()
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        deepseek_client=deepseek_client,
        source_name=config.source_name
    )
    # Run the agent without streaming
    result = await documentation_expert.run(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1]
    )
    
    # Extract all new messages using the correct method
    new_messages = list(result.new_messages())
    
    # Find the last response message and extract its text
    response_text = ""
    for msg in reversed(new_messages):
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    response_text = part.content
                    break
            break

    # Display the response
    st.markdown(response_text)

    # Add new messages to session state
    st.session_state.messages.extend(new_messages)


async def main():
    config = CrawlerConfig()
    st.title(config.ui_title)
    st.write(config.ui_description)

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about the Documentation?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's response
        with st.chat_message("assistant"):
            # Run the agent and display the response
            await run_agent_and_display(user_input)


if __name__ == "__main__":
    asyncio.run(main())
