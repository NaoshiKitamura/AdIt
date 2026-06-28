# AdIt - Lab AI Assistant

A CLI-based AI assistant that answers questions, creates files, and edits code
using RAG over lab documents (LAMMPS manuals, research reports, etc.).

## Stack

- **LLM**: Ollama + qwen3.5:9b
- **RAG**: ChromaDB + HuggingFace multilingual-e5-small
- **API**: FastAPI (OpenAI-compatible)
- **CLI**: Aider

## For Administrators: Installation

### 1. Clone the repository

    git clone https://github.com/NaoshiKitamura/AdIt.git /opt/AdIt

### 2. Create conda environment

    conda create -n AdIt python=3.12
    conda activate AdIt
    pip install fastapi uvicorn httpx langchain-chroma langchain-huggingface \
                langchain-community sentence-transformers beautifulsoup4 \
                lxml pypdf aider-chat

### 3. Build the RAG database

Place documents (HTML, PDF, TXT) in /opt/AdIt/documents/, then run:

    cd /opt/AdIt
    python scripts/build_db.py

### 4. Install the command

    chmod +x /opt/AdIt/AdIt
    ln -s /opt/AdIt/AdIt /usr/local/bin/AdIt

## For Users: Usage

### Start

    AdIt

The RAG server starts automatically and the chat session begins.

### Examples

    > How do I use the fix nvt command?
    > Create an NVT simulation input file for silicon.
    > What does this error mean?

### Chat modes

- ask mode (default): question answering
- code mode: file creation and editing (switch with /code)

## Adding Documents

Place HTML, PDF, or text files in documents/ and rebuild the database:

    cd /opt/AdIt
    python scripts/build_db.py
