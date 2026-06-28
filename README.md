# AdIt - Lab AI Assistant
A CLI-based AI assistant that answers questions, creates files, and edits code
using RAG over lab documents (LAMMPS manuals, research reports, etc.).

## Stack
- **LLM**: Ollama (default: qwen3.5:9b, switchable — see [Switching LLM Models](#switching-llm-models))
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
    python -m scripts.build_db

To rebuild from scratch:

    python -m scripts.build_db --rebuild

### 5. Install the command
    chmod +x /opt/AdIt/AdIt
    ln -s /opt/AdIt/AdIt /usr/local/bin/AdIt

## Adding Documents

Place HTML, PDF, TXT, DOCX, or PPTX files in `documents/` and update the database:

    conda activate AdIt
    python -m scripts.build_db

### Example: LAMMPS Documentation

Download the LAMMPS tarball and extract the HTML manual into `documents/lammps_html/`:

```bash
# Download the latest stable tarball
wget https://download.lammps.org/tars/lammps-stable.tar.gz

# Extract only the HTML documentation
tar -xzf lammps-stable.tar.gz --wildcards '*/doc/html/*' --strip-components=3 -C documents/lammps_html/

# Add to RAG database
conda activate AdIt
python -m scripts.build_db
```

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

## Switching LLM Models

AdIt allows you to switch the LLM model depending on your use case and required response speed.

Edit `rag/config.py`:

```python
OLLAMA_CHAT_MODEL = "your-model-name"
```

Then pull the model with Ollama before starting:

```bash
ollama pull your-model-name
```

For available models, search the [Ollama model library](https://ollama.com/library).

### Example
```bash
# Switch to a lighter model for faster responses
ollama pull qwen2.5:3b

# Switch to a larger model for higher quality
ollama pull qwen2.5:27b
```

## Tuning RAG Performance

All parameters below are defined in `rag/config.py`.

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `SEARCH_K` | `8` | Number of chunks retrieved from ChromaDB |
| `TOP_K` | `3` | Number of chunks kept after reranking |
| `PAGE_CONTENT_LIMIT` | `1500` | Max characters per chunk sent to reranker |
| `CONTEXT_MAX_CHARS` | `1200` | Max characters of context passed to the LLM |

Reducing `SEARCH_K` and `TOP_K` speeds up responses. Increasing them may improve answer quality.

### Chunking

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Number of characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |

After changing chunking parameters, run:

```bash
conda activate AdIt
python -m scripts.build_db
```

The updated `build_db.py` automatically detects chunk config changes and re-embeds only the affected files.
