# utils/text.py
import re
from typing import List

def split_sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", t)
    return [p.strip() for p in parts if p.strip()]
