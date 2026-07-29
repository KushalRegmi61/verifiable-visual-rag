"""SQLAlchemy 2.0 tables.

Deliberately separate from contracts.py. The wire format and the storage schema
must be able to evolve independently, which is precisely the coupling that
SQLModel's single-class approach would have introduced.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class UtcDateTime(TypeDecorator):
    """A timestamp that is always aware and always UTC, on every backend.

    `DateTime(timezone=True)` alone is not enough on SQLite, and the failure is
    worse than a lost tzinfo. SQLite has no native datetime type, so SQLAlchemy
    binds the value through a string format that **discards the offset**, then
    stores the remaining wall-clock digits. Storing 12:00+05:45 therefore writes
    '2026-01-01 12:00:00.000000' and reads back a naive 12:00, when the instant
    was actually 06:15 UTC. The timestamp is silently shifted, not merely
    stripped, and nothing raises. Postgres stores it correctly, which is exactly
    what makes the divergence dangerous: it corrupts only in development.

    Binding normalizes to UTC first, so the wall clock that reaches the column
    is already the right instant; loading re-attaches UTC when the backend hands
    back a naive value. Naive input is rejected outright rather than guessed at,
    since assuming a zone is how the shift gets reintroduced. The emitted DDL is
    unchanged, so this is a Python-side coercion only.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("refusing to store a naive datetime; pass an aware one")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    n_pages: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))

    pages: Mapped[list["Page"]] = relationship(back_populates="document")


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_sha: Mapped[str] = mapped_column(ForeignKey("documents.sha256"))
    page_no: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str] = mapped_column(Text)
    width_px: Mapped[int] = mapped_column(Integer)
    height_px: Mapped[int] = mapped_column(Integer)
    dpi: Mapped[int] = mapped_column(Integer)

    document: Mapped[Document] = relationship(back_populates="pages")
    boxes: Mapped[list["Box"]] = relationship(back_populates="page")

    __table_args__ = (Index("ix_pages_doc_page", "doc_sha", "page_no", unique=True),)


class Box(Base):
    """One candidate box, normalized to 0-1. Words only; coarser boxes derive."""

    __tablename__ = "boxes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id"))
    kind: Mapped[str] = mapped_column(String(16))
    x0: Mapped[float] = mapped_column(Float)
    y0: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text, default="")
    block_no: Mapped[int] = mapped_column(Integer, default=-1)
    line_no: Mapped[int] = mapped_column(Integer, default=-1)
    word_no: Mapped[int] = mapped_column(Integer, default=-1)

    page: Mapped[Page] = relationship(back_populates="boxes")

    __table_args__ = (Index("ix_boxes_page", "page_id"),)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Named explicitly: SQLite can only add this constraint via batch mode, and
    # batch mode refuses an anonymous constraint ("Constraint must have a name").
    doc_sha: Mapped[str] = mapped_column(
        ForeignKey("documents.sha256", name="fk_jobs_doc_sha_documents")
    )
    stage: Mapped[str] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(16))
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))

    # Jobs are the audit log for documents, so an orphan job row should not be
    # possible; doc_sha is also the only column anyone filters this table on.
    __table_args__ = (Index("ix_jobs_doc_sha", "doc_sha"),)
