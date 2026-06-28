# AdIt - Lab AI Assistant

A CLI-based AI assistant that answers questions, creates files, and edits code
using RAG over lab documents (LAMMPS manuals, research reports, etc.).

## Stack

* **LLM**: Ollama (default: qwen3.5:9b, switchable — see [Switching LLM Models](#switching-llm-models))
* **RAG**: ChromaDB + HuggingFace multilingual-e5-small
* **API**: FastAPI (OpenAI-compatible)
* **CLI**: Aider

## Supported Document Formats

* HTML / HTM
* PDF
* TXT
* DOCX (Word)
* PPTX (PowerPoint)

## For Administrators: Installation

### 1. Clone the repository

```bash
git clone https://github.com/NaoshiKitamura/AdIt.git /opt/AdIt
```

### 2. Create the conda environment

```bash
conda create -n AdIt python=3.12 -y
conda activate AdIt

pip install fastapi uvicorn httpx langchain-chroma langchain-huggingface \
            langchain-community sentence-transformers beautifulsoup4 \
            lxml pypdf aider-chat python-docx python-pptx
```

### 3. Install the command

```bash
chmod +x /opt/AdIt/AdIt
sudo ln -s /opt/AdIt/AdIt /usr/local/bin/AdIt
```

> `sudo` is typically required because `/usr/local/bin` is owned by the system.

### 4. Add documents

Place HTML, PDF, TXT, DOCX, or PPTX files in:

```text
/opt/AdIt/documents/
```

Then build the RAG database:

```bash
conda activate AdIt
python -m scripts.build_db
```

To rebuild the database from scratch:

```bash
python -m scripts.build_db --rebuild
```

#### Example: LAMMPS Documentation

The LAMMPS HTML manual is a convenient dataset for testing AdIt.

```bash
git clone https://github.com/lammps/lammps.git
cd lammps/doc
make html
```

Copy the generated HTML files into AdIt's document directory:

```bash
mkdir -p /opt/AdIt/documents/lammps_html
cp -r html/* /opt/AdIt/documents/lammps_html/
```

Then rebuild the RAG database:

```bash
conda activate AdIt
python -m scripts.build_db
```

---

## For Users: Usage

### Activate the environment and start AdIt

```bash
conda activate AdIt
AdIt
```

The RAG server starts automatically and the chat session begins.

### Examples

```text
> How do I use the fix nvt command?
> Create an NVT simulation input file for silicon.
> What does this error mean?
```

### Chat modes

* **ask mode** (default): Question answering
* **code mode**: File creation and editing (switch with `/code`)

---

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

### Examples

```bash
# Switch to a lighter model for faster responses
ollama pull qwen2.5:3b

# Switch to a larger model for higher quality
ollama pull qwen2.5:27b
```

---

## Tuning RAG Performance

All parameters below are defined in `rag/config.py`.

### Retrieval

| Variable             | Default | Description                                       |
| -------------------- | ------- | ------------------------------------------------- |
| `SEARCH_K`           | `8`     | Number of chunks retrieved from ChromaDB          |
| `TOP_K`              | `3`     | Number of chunks kept after reranking             |
| `PAGE_CONTENT_LIMIT` | `1500`  | Maximum characters per chunk sent to the reranker |

