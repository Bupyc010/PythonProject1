from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


class Photo(Base):
    __tablename__ = "photo"

    id = Column(Integer, primary_key=True)
    photo = Column(String)
    photo_date = Column(DateTime)
    user_id = Column(Integer, ForeignKey("user.id"))

    user = relationship('User', back_populates='photos')
    code = relationship("Code", back_populates="photo")