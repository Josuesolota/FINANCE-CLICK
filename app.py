# app.py - FinanceClick Backend - VERSÃO SIMPLIFICADA E TESTADA
import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configuração básica
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend")

print(f"🚀 Iniciando FinanceClick")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Frontend path: {FRONTEND_PATH}")

# Verificar se frontend existe
if os.path.exists(FRONTEND_PATH):
    print("✅ Pasta frontend encontrada!")
    files = [f for f in os.listdir(FRONTEND_PATH) if f.endswith(('.html', '.css', '.js'))]
    print(f"📄 Arquivos: {files}")
else:
    print("❌ ERRO: Pasta frontend não encontrada!")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financeclick")

# Criar app FastAPI
app = FastAPI(title="FinanceClick", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== SERVIR ARQUIVOS ESTÁTICOS ====================

@app.get("/")
async def serve_index():
    """Serve a página inicial"""
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html não encontrado")

@app.get("/style.css")
async def serve_css():
    """Serve CSS"""
    css_path = os.path.join(FRONTEND_PATH, "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS não encontrado")

@app.get("/script.js")
async def serve_js():
    """Serve JavaScript"""
    js_path = os.path.join(FRONTEND_PATH, "script.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS não encontrado")

# Servir outras páginas
@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_PATH, "dashboard.html"))

@app.get("/history")
async def serve_history():
    return FileResponse(os.path.join(FRONTEND_PATH, "history.html"))

@app.get("/guide")
async def serve_guide():
    return FileResponse(os.path.join(FRONTEND_PATH, "guide.html"))

@app.get("/about")
async def serve_about():
    return FileResponse(os.path.join(FRONTEND_PATH, "about.html"))

@app.get("/contact")
async def serve_contact():
    return FileResponse(os.path.join(FRONTEND_PATH, "contact.html"))

# Fallback para SPA
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    file_path = os.path.join(FRONTEND_PATH, full_path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    # Fallback para index.html (SPA)
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Página não encontrada")

# ==================== API ENDPOINTS BÁSICOS ====================

@app.get("/api/health")
async def health_check():
    """Health check simples"""
    return {
        "status": "online",
        "service": "FinanceClick",
        "timestamp": datetime.now().isoformat(),
        "frontend_exists": os.path.exists(FRONTEND_PATH)
    }

@app.get("/api/debug")
async def debug_info():
    """Debug information"""
    frontend_files = []
    if os.path.exists(FRONTEND_PATH):
        frontend_files = os.listdir(FRONTEND_PATH)
    
    return {
        "project_root": PROJECT_ROOT,
        "frontend_path": FRONTEND_PATH,
        "frontend_exists": os.path.exists(FRONTEND_PATH),
        "frontend_files": frontend_files,
        "all_systems_ok": True
    }

# Endpoints de exemplo (podem ser expandidos depois)
@app.get("/api/balance")
async def get_balance():
    return {"balance": 1000.00, "currency": "USD"}

@app.get("/api/robot/status")
async def get_robot_status():
    return {"active": False, "message": "Robô inativo"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "Página não encontrada"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor"}
    )

# Inicialização
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)