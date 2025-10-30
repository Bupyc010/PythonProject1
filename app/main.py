from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.database import create_db_and_tables
from app.models.photo import routers as photo_routers
from app.models.user import routers as user_routers


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Приложение запускается. Создаем базу данных...")
    await create_db_and_tables()
    print("База данных инициализирована.")
    yield
    print("Приложение завершает работу.")


app = FastAPI(
    title="Нейросеть",
    lifespan=lifespan
)

app.include_router(photo_routers.router)
app.include_router(user_routers.router)


@app.get("/")
async def root():
    return {"message": "Это проект по нейросетям"}