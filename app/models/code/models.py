from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from app.core.database import Base


class Code(Base):
    __tablename__ = "code"

    id = Column(Integer, primary_key=True)
    code = Column(String)
    code_date = Column(DateTime)
    id_photo = Column(Integer, ForeignKey("photo.id"))

    photo = relationship("Photo", back_populates="code")