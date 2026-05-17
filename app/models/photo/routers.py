from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import aiofiles

from app.core.dependencies import get_async_db
from app.models.photo import crud as photo_crud
from app.models.code import crud as code_crud
from app.models.services import code


router = APIRouter(
    prefix="/photo"
)

#Получить список всех схем.
@router.get("/")
async def read_code(db: AsyncSession = Depends(get_async_db)):
    return await photo_crud.get_photo(db)


#Асинхронно сохраняет загруженный файл на диск, читая его по частям.
@router.post("/")
async def create_upload_file_async_save(file: UploadFile = File(...), db: AsyncSession = Depends(get_async_db)):
    file_location = f"app/photo/{file.filename}"

    try:
        # Асинхронное сохранение файла
        async with aiofiles.open(file_location, "wb") as out_file:
            chunk_size = 1024 * 1024
            while content := await file.read(chunk_size):
                await out_file.write(content)

        # Работа с БД
        await photo_crud.create_photo(db, file_location)
        photo_id = await photo_crud.get_id_photo(db, file_location)

        # Генерация кода
        photo_code = code.cod(file_location)
        await code_crud.create_code(db, photo_code, photo_id)

        # Возвращаем только сгенерированный код
        return {
            "photo_code": photo_code
        }
    except Exception as e:
        return {"error": f"Could not save file: {e}"}