from typing import Optional, Type, TypeVar

from fastapi import Body, Depends, FastAPI, Query, Request

from ._base import BaseModel, Controller, DatabaseModel, Repository, Surreal, get_db

T = TypeVar("T", bound=BaseModel)
M = TypeVar("M", bound=DatabaseModel)


def create_controller(schema: Type[T], model: Type[M]) -> Controller[M, T]:
    app = Controller[M, T](model=model, repository=Repository[M, T](model=model))

    @app.get("/{namespace}/{key}")
    async def _(
        request: Request, id: Optional[str] = Query(None), db: Surreal = Depends(get_db)
    ):
        query_params = dict(request.query_params)
        return await app.repository.read_(where=query_params, db=db, id=id)

    @app.post("/{namespace}/{key}")
    async def _(data: schema, db: Surreal = Depends(get_db)):
        return await app.repository.create_(data=data, db=db)

    @app.put("/{namespace}/{key}")
    async def _(
        id: str = Query(...), data: model = Body(...), db: Surreal = Depends(get_db)
    ):
        return await app.repository.update_(data=data, db=db, id=id)

    @app.delete("/{namespace}/{key}")
    async def _(id: str = Query(...), db: Surreal = Depends(get_db)):
        return await app.repository.delete_(db=db, id=id)

    return app


class Application(FastAPI):
    def add(self, model: Type[M], schema: Type[T]) -> FastAPI:
        controller = create_controller(schema=schema, model=model)
        self.include_router(controller, prefix="/api")
        return self
