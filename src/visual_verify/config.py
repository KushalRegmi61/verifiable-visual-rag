"""Environment-driven settings.

No module anywhere in this package hardcodes a connection string or a path.
That is what makes SQLite to Neon, and local Qdrant to Qdrant Cloud, an env
var change rather than a code change.

Stdlib only, so importing this never pulls a settings library into the core.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_URL = "sqlite:///data/index.db"
DEFAULT_DATA_DIR = "data"
DEFAULT_RENDER_DPI = 150
DEFAULT_TEXT_PAGE_RATIO = 0.6


@dataclass(frozen=True)
class Settings:
    db_url: str = DEFAULT_DB_URL
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    render_dpi: int = DEFAULT_RENDER_DPI
    min_text_page_ratio: float = DEFAULT_TEXT_PAGE_RATIO
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    reader_provider: str = "openai"
    reader_model: str = "gpt-4o"
    verifier_provider: str = "google"
    verifier_model: str = "gemini-2.0-flash"
    abstain_threshold: float = 6.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_url=os.getenv("VVRAG_DB_URL", DEFAULT_DB_URL),
            data_dir=Path(os.getenv("VVRAG_DATA_DIR", DEFAULT_DATA_DIR)),
            render_dpi=int(os.getenv("VVRAG_RENDER_DPI", DEFAULT_RENDER_DPI)),
            min_text_page_ratio=float(
                os.getenv("VVRAG_MIN_TEXT_PAGE_RATIO", DEFAULT_TEXT_PAGE_RATIO)
            ),
            qdrant_url=os.getenv("VVRAG_QDRANT_URL"),
            qdrant_api_key=os.getenv("VVRAG_QDRANT_API_KEY"),
            reader_provider=os.getenv("VVRAG_READER_PROVIDER", "openai"),
            reader_model=os.getenv("VVRAG_READER_MODEL", "gpt-4o"),
            verifier_provider=os.getenv("VVRAG_VERIFIER_PROVIDER", "google"),
            verifier_model=os.getenv("VVRAG_VERIFIER_MODEL", "gemini-2.0-flash"),
            abstain_threshold=float(os.getenv("VVRAG_ABSTAIN_THRESHOLD", "6.0")),
        )

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def agent_cache_dir(self) -> Path:
        return self.data_dir / "agent_cache"
