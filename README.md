# Agentic AI - LangChain Fundamentals

A comprehensive guide covering LangChain concepts for building agentic AI applications.

## Contents

The `updatedlangchain/` directory contains Jupyter notebooks demonstrating:

- **Model Integration** (`2-modelintegration.ipynb`) - Connecting to various LLM providers (OpenAI, Groq, Google GenAI)
- **Tools** (`3-tools.ipynb`) - Adding capabilities to LLM agents
- **Messages** (`4-messages.ipynb`) - Working with conversation messages
- **Structured Output** (`5-structuredoutput.ipynb`) - Getting typed responses using Pydantic models and TypedDict

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Configure your API keys
```

### Required Environment Variables

```env
CHAT_API_BASE_URL=your_url
CHAT_API_KEY=your_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Dependencies

- langchain
- langchain_community
- langchain-openai
- langchain-groq
- langchain-google-genai
- python-dotenv
- ipykernel

## Running

```bash
# Launch Jupyter
jupyter notebook
```

## Key Concepts

### Structured Output with Pydantic

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(..., description="The title of the movie")
    director: str = Field(..., description="The director of the movie")
    release_year: int = Field(..., description="The release year")
    ratings: float = Field(..., description="Rating out of 10")

model_with_structure = custom_llm.with_structured_output(Movie)
```

### Structured Output with TypedDict

```python
from typing import TypedDict, Annotated

class MovieDict(TypedDict):
    title: Annotated[str, ..., "The title of the movie"]
    director: Annotated[str, ..., "The director of the movie"]
```
