"""Example domain objects for testing."""
from __future__ import annotations

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from starlite_saqlalchemy import db


class Parent(db.orm.Base):
    name: Mapped[str]
    children: Mapped[list[Child]] = relationship("Chil", back_populate="parent")
