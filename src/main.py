from fastapi import FastAPI
from .database import engine
from .models import Base
from .routers import home, chatbot, ws, history, auth


def create_app() -> FastAPI:
    app = FastAPI()
    Base.metadata.create_all(bind=engine)
    # Mount the routers
    app.include_router(auth.router)    # route: "/history"
    app.include_router(home.router)             # routes: "/" and "/healthy"
    app.include_router(chatbot.router, prefix="")  # route: "/chatbot"
    app.include_router(ws.router, prefix="")    # route: "/ws"
    app.include_router(history.router, prefix="")    # route: "/history"

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
