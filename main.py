from fastapi.staticfiles import StaticFiles

from prisma import Prisma
from src import create_app

app = create_app()
app.mount("/static", StaticFiles(directory="upload"), name="static")
db = Prisma(auto_register=True, log_queries=True)


@app.on_event("startup")  # type: ignore
async def startup():
    await db.connect()


@app.on_event("shutdown")  # type: ignore
async def shutdown():
    await db.disconnect()
