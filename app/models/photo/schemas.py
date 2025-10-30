from pydantic import BaseModel
from pydantic import ConfigDict
from datetime import date

class PhotoBase(BaseModel):
    photo: str
    photo_date: date
    user_id: int


class Photo(PhotoBase):
    id: int

    model_config = ConfigDict(from_attributes=True)