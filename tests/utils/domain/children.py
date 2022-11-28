"""Example domain objects for testing."""
from __future__ import annotations

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from starlite_saqlalchemy import db


class Child(db.orm.Base):
    name: Mapped[str]
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent.id"))
    parent: Mapped[Parent] = relationship("Parent", back_populates="children")
