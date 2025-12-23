import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.t_nexus.backend.database import engine, Base
from src.t_nexus.backend.routers import (
    auth_router,
    overview_router,
    metrics_router,
    manual_review_router,
    notifications_router,
    rag_router,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="T-Nexus",
    description="Customer Support Automation Dashboard API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(overview_router.router)
app.include_router(metrics_router.router)
app.include_router(manual_review_router.router)
app.include_router(notifications_router.router)
app.include_router(rag_router.router)

frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "t_nexus/frontend")

@app.get("/")
async def root():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "T-Nexus API", "docs": "/docs"}

@app.get("/login")
async def login_page():
    login_path = os.path.join(frontend_path, "login.html")
    if os.path.exists(login_path):
        return FileResponse(login_path)
    return {"message": "Login page"}

@app.get("/register")
async def register_page():
    register_path = os.path.join(frontend_path, "register.html")
    if os.path.exists(register_path):
        return FileResponse(register_path)
    return {"message": "Register page"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "T-Nexus"}

@app.get("/api/bot/stats")
async def bot_stats():
    from src.t_nexus.backend.telegram.tg_database import TelegramDatabase
    from src.t_nexus.backend.config import settings
    
    try:
        db = TelegramDatabase(settings.TG_DB_PATH)
        return {
            "total_conversations": db.get_total_conversations(),
            "total_messages": db.get_total_messages(),
            "feedback_stats": db.get_feedback_stats()
        }
    except Exception as e:
        return {"error": str(e)}

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
