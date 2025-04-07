import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def export_chunks_to_json(
        input_text: str,
        chunks: List[str],
        semantic_embed_model: Optional[str] = None,
        output_dir: str = "resources/chunking"
) -> str:
    """
    Saves the original text, chunks, and optionally the semantic embedding model to a timestamped JSON file.

    Args:
        input_text (str): The original full text.
        chunks (List[str]): The resulting list of chunks.
        semantic_embed_model (Optional[str]): The name of the embedding model used.
        output_dir (str): Directory to save the output file.

    Returns:
        str: Path to the written JSON file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"text_chunks_{timestamp}.json"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / filename

    payload = {
        "input_text": input_text,
        "chunks": chunks
    }

    if semantic_embed_model:
        payload["semantic_embed_model"] = semantic_embed_model

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(filepath)
