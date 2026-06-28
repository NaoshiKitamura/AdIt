# AdIt - Lab AI Assistant

A CLI-based AI assistant that answers questions, creates files, and edits code
using RAG over lab documents (LAMMPS manuals, research reports, etc.).

## Stack

- **LLM**: Ollama + qwen3.5:9b
- **RAG**: ChromaDB + HuggingFace multilingual-e5-small
- **API**: FastAPI (OpenAI-compatible)
- **CLI**: Aider

## Supported Document Formats

- HTML / HTM
- PDF
- TXT
- DOCX (Word)
- PPTX (PowerPoint)

## For Administrators: Installation

### 1. Clone the repository

    git clone https://github.com/NaoshiKitamura/AdIt.git /opt/AdIt

### 2. Create conda environment

    conda create -n AdIt python=3.12 -y
    conda activate AdIt
    pip install fastapi uvicorn httpx langchain-chroma langchain-huggingface \
                langchain-community sentence-transformers beautifulsoup4 \
                lxml pypdf aider-chat python-docx python-pptx

### 3. Add documents

Place documents (HTML, PDF, TXT, DOCX, PPTX) in /opt/AdIt/documents/

### 4. Build the RAG database

    conda activate AdIt
    cd /opt/AdIt
    python -m scripts.build_db

To rebuild from scratch:

    python -m scripts.build_db --rebuild

### 5. Install the command

    chmod +x /opt/AdIt/AdIt
    ln -s /opt/AdIt/AdIt /usr/local/bin/AdIt

## For Users: Usage

### Activate environment and start

    conda activate AdIt
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

Place HTML, PDF, TXT, DOCX, or PPTX files in documents/ and update the database:

    conda activate AdIt
    cd /opt/AdIt
    python -m scripts.build_db

To rebuild the entire database from scratch:

    python -m scripts.build_db --rebuild
