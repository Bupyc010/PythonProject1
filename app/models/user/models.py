from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    photos = relationship("Photo", back_populates="user")