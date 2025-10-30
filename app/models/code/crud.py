from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.code.models import Code
from datetime import date


#добовление новой преобразованной схемы
async def create_code(db: AsyncSession, code: str, id_photo: int):
    db_code = Code(code = code, code_date = date.today(), id_photo = id_photo)
    db.add(db_code)
    await db.commit()
    await db.refresh(db_code)
    return db_code
