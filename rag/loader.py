# rag/loader.py

from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


# ==========================================================
# HTML loader
# ==========================================================

def load_html(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # Remove unnecessary elements
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    title = (
        soup.title.text.strip()
        if soup.title
        else str(file_path)
    )

    body = soup.body

    if body:
        text = body.get_text(separator="\n")
    else:
        text = soup.get_text(separator="\n")

    text = "\n".join(
        line.strip()
        for line in text.split("\n")
        if line.strip()
    )

    return Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "title": title,
            "type": "html",
        },
    )


# ==========================================================
# PDF loader
# ==========================================================

def load_pdf(file_path):

    loader = PyPDFLoader(str(file_path))

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["type"] = "pdf"

    return docs


# ==========================================================
# TXT loader
# ==========================================================

def load_txt(file_path):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8",
    )

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["type"] = "txt"

    return docs


# ==========================================================
# Generic loader
# ==========================================================

def load_file(file_path):

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    if suffix in (".html", ".htm"):
        return [load_html(file_path)]

    if suffix == ".txt":
        return load_txt(file_path)

    return []
