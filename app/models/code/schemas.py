from pydantic import BaseModel
from pydantic import ConfigDict
from datetime import date


class CodeBase(BaseModel):
    code: str
    code_date: date
    id_photo: int


class Code(CodeBase):
    id: int

    model_config = ConfigDict(from_attributes=True)