from pydantic import BaseModel, Field, EmailStr
from pydantic import ConfigDict
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="Email пользователя")
    password: str = Field(..., min_length=8, description="Пароль (минимум 8 символов)")


class User(UserCreate):
    id: int
    email: EmailStr
    is_active: bool
    password: Optional[str] = Field(None, exclude=True)

    model_config = ConfigDict(from_attributes=True)