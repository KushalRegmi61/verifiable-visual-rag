"""SQLAlchemy 2.0 tables.

Deliberately separate from contracts.py. The wire format and the storage schema
must be able to evolve independently, which is precisely the coupling that
SQLModel's single-class approach would have introduced.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    n_pages: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

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
    doc_sha: Mapped[str] = mapped_column(String(64))
    stage: Mapped[str] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(16))
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )
