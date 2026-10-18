def format_docs(docs: list) -> str:
    """
    Format list of Documents into a single context string.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get('url', 'N/A')
        title = doc.metadata.get('title', 'N/A')
        content = doc.page_content[:500]  # First 500 chars
        formatted.append(
            f"[Source {i}: {url}]\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
        )
    return "\n---\n".join(formatted)
