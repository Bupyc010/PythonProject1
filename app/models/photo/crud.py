from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.photo.models import Photo
from app.models.photo.schemas import PhotoBase, Photo as PhotoShema
from app.models.code.models import Code
from app.models.code.schemas import CodeBase, Code as CodeShema
from datetime import date

# Получить ID схему по имени
async def get_id_photo(db: AsyncSession, name: str):
    result = await db.scalar(select(Photo.id).where(Photo.photo == name))
    return result

#Получить список всех схем
async def get_photo(db: AsyncSession):
    stmt = select(Photo.photo, Code.code).select_from(Photo).join(Code, Photo.id == Code.id_photo)
    scalar_result = await db.execute(stmt)
    result = scalar_result.all()
    data = []
    for row in result:
        photo_value = row[0] if row[0] else "No photo"  # Обработка None
        code_value = row[1] if row[1] else "No code"
        data.append({"photo": photo_value, "code": code_value})
    return data

#добовление новой схемы
async def create_photo(db: AsyncSession, photo: str):
    db_photo = Photo(photo = photo, photo_date = date.today())
    db.add(db_photo)
    await db.commit()
    await db.refresh(db_photo)
    return db_photo

