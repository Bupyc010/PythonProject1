from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import aiofiles

from app.core.dependencies import get_async_db
from app.models.photo import crud as photo_crud
from app.models.code import crud as code_crud


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
        # Открываем файл асинхронно с помощью aiofiles
        async with aiofiles.open(file_location, "wb") as out_file:
            # Читаем файл по частям (чанками), например, по 1 МБ
            chunk_size = 1024 * 1024
            while content := await file.read(chunk_size):  # Асинхронное чтение из UploadFile
                await out_file.write(content)  # Асинхронная запись чанка в файл

        await photo_crud.create_photo(db, file_location)
        photo_id = await photo_crud.get_id_photo(db, file_location)
        await code_crud.create_code(db,"Файл загружен", photo_id)
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        return {"error": f"Could not save file: {e}"}