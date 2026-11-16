"""HTTP interface for the local news RAG."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from .rag import RAG, Ask, AskRequest, RAGError


def create_app(rag: RAG | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = rag if rag is not None else RAG.from_env()
        yield

    app = FastAPI(title="News RAG", lifespan=lifespan)

    @app.get("/health")
    def health(response: Response):
        result = app.state.rag.health()
        if not result["ready"]:
            response.status_code = 503
        return result

    @app.post("/ask")
    def ask(request: AskRequest):
        try:
            return app.state.rag.ask(
                Ask(question=request.question, as_of=request.as_of, period=request.period)
            )
        except RAGError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    return app


app = create_app()
