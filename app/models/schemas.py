from typing import List, Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    title: str
    summary: str
    short_summary: str
    tags: List[str]
