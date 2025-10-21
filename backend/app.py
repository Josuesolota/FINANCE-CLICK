# app.py - FinanceClick Backend with Accumulator Options AI Robot
# VERSÃO PRODUÇÃO - CONEXÃO OTIMIZADA PARA RENDER
import os
import json
import asyncio
import pickle
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import functools
import time

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, validator
import secrets

# Deriv API import
from deriv_api import DerivAPI

# ==================== CONFIGURAÇÃO PRODUÇÃO ====================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend")

print(f"🚀 Iniciando FinanceClick em PRODUÇÃO")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Frontend path: {FRONTEND_PATH}")

# Configure logging para produção
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("financeclick")

# Load environment variables
load_dotenv()

# Configuração Produção
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_REDIRECT_URL = os.getenv("DERIV_REDIRECT_URL", "https://finance-click.onrender.com/auth/callback")
DERIV_API_URL = os.getenv("DERIV_API_URL", "wss://ws.deriv.com/websockets/v3")
PORT = int(os.getenv("PORT", "10000"))

# Security settings produção
ALLOWED_ORIGINS = ["*"]
SESSION_TIMEOUT = 3600

# Variáveis globais
deriv_service = None
active_tokens = {}
user_sessions = {}
robot_active = False
robot_tasks = {}
contact_messages = []

# Knowledge base padrão
DEFAULT_KNOWLEDGE_BASE = {
    "regras": [
        {
            "keywords": ["accumulator", "accumulators", "accumulator options"],
            "resposta": "Accumulator Options são instrumentos financeiros que permitem lucrar com mercados laterais através de crescimento composto. Escolha entre 1% e 5% de taxa de crescimento."
        }
    ]
}

# Cache para produção
class SimpleCache:
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str) -> Any:
        if key in self._cache:
            data, expiry = self._cache[key]
            if time.time() < expiry:
                return data
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, expire: int = 60):
        self._cache[key] = (value, time.time() + expire)
    
    def clear(self):
        self._cache.clear()

simple_cache = SimpleCache()

def cache(expire: int = 60):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cached_result = simple_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            result = await func(*args, **kwargs)
            simple_cache.set(cache_key, result, expire)
            return result
        return wrapper
    return decorator

# SERVIÇO DERIV API OTIMIZADO PARA PRODUÇÃO
class DerivAPIService:
    def __init__(self):
        self.api = None
        self.connected = False
        self.connection_attempts = 0
        self.max_retries = 3
        self.retry_delay = 5

    async def connect(self):
        """✅ CONEXÃO ROBUSTA PARA PRODUÇÃO COM RETRY"""
        while self.connection_attempts < self.max_retries and not self.connected:
            try:
                logger.info(f"🔗 Tentativa {self.connection_attempts + 1} de conexão com Deriv API...")
                
                # Conexão com timeout
                self.api = DerivAPI(
                    app_id=DERIV_APP_ID,
                    endpoint=DERIV_API_URL
                )
                
                # Teste de conexão com timeout
                try:
                    ping_response = await asyncio.wait_for(
                        self.api.ping({"ping": 1}),
                        timeout=10.0
                    )
                    self.connected = True
                    self.connection_attempts = 0
                    logger.info("✅ Conectado à Deriv API em produção")
                    return True
                    
                except asyncio.TimeoutError:
                    logger.warning("⏰ Timeout na conexão com Deriv API")
                    raise Exception("Timeout na conexão")
                    
            except Exception as e:
                self.connection_attempts += 1
                self.connected = False
                logger.error(f"❌ Falha na conexão Deriv API (tentativa {self.connection_attempts}): {e}")
                
                if self.connection_attempts < self.max_retries:
                    logger.info(f"🔄 Nova tentativa em {self.retry_delay} segundos...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error("💥 Todas as tentativas de conexão falharam")
                    return False
        
        return self.connected

    async def ensure_connection(self):
        """✅ GARANTIR CONEXÃO ATIVA EM PRODUÇÃO"""
        if not self.connected or not self.api:
            return await self.connect()
        
        try:
            # Verificar se a conexão ainda está ativa
            await asyncio.wait_for(
                self.api.ping({"ping": 1}),
                timeout=5.0
            )
            return True
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("🔌 Conexão perdida, reconectando...")
            self.connected = False
            return await self.connect()

    async def authorize(self, token: str) -> Optional[Dict]:
        """✅ AUTORIZAÇÃO ROBUSTA PARA PRODUÇÃO"""
        if not await self.ensure_connection():
            return None
            
        try:
            response = await self.api.authorize({"authorize": token})
            if response and 'authorize' in response:
                logger.info(f"🔐 Autorizado: {response['authorize'].get('loginid', 'Unknown')}")
                return response
            else:
                logger.warning("❌ Autorização falhou - resposta inválida")
                return None
        except Exception as e:
            logger.error(f"❌ Erro na autorização: {e}")
            return None

    async def get_balance(self, token: str) -> Optional[float]:
        """✅ SALDO COM FALLBACK PARA PRODUÇÃO"""
        try:
            auth_data = await self.authorize(token)
            if auth_data and 'authorize' in auth_data:
                balance = float(auth_data['authorize']['balance'])
                logger.info(f"💰 Saldo obtido: {balance}")
                return balance
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
        
        return None

    async def production_health_check(self):
        """✅ HEALTH CHECK ESPECÍFICO PARA PRODUÇÃO"""
        try:
            if await self.ensure_connection():
                ping = await self.api.ping({"ping": 1})
                return {
                    "status": "healthy",
                    "deriv_connected": True,
                    "ping_response": ping is not None
                }
            else:
                return {
                    "status": "unhealthy", 
                    "deriv_connected": False,
                    "error": "Não foi possível conectar à Deriv API"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "deriv_connected": False,
                "error": str(e)
            }

# LIFESPAN OTIMIZADO PARA PRODUÇÃO
@asynccontextmanager
async def lifespan(app: FastAPI):
    global deriv_service
    
    logger.info("🏁 Iniciando FinanceClick em ambiente de PRODUÇÃO")
    
    # Inicializar serviço com conexão robusta
    deriv_service = DerivAPIService()
    
    # Conexão em background com retry automático
    async def production_connect():
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts and not deriv_service.connected:
            try:
                success = await deriv_service.connect()
                if success:
                    logger.info("🎯 Conexão Deriv API estabelecida em produção")
                    break
                else:
                    attempt += 1
                    if attempt < max_attempts:
                        wait_time = attempt * 10  # Backoff exponencial
                        logger.info(f"🔄 Nova tentativa de conexão em {wait_time} segundos...")
                        await asyncio.sleep(wait_time)
            except Exception as e:
                attempt += 1
                logger.error(f"💥 Erro na tentativa {attempt}: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(10)
        
        if not deriv_service.connected:
            logger.error("💥 Falha crítica: Não foi possível conectar à Deriv API")

    # Iniciar conexão em background sem bloquear
    asyncio.create_task(production_connect())
    
    logger.info(f"✅ Serviço de produção inicializado na porta {PORT}")
    yield
    
    logger.info("🔴 Encerrando serviço de produção...")

# APP FASTAPI PARA PRODUÇÃO
app = FastAPI(
    title="FinanceClick AI Trading Platform",
    description="Backend with Accumulator Options AI Robot - PRODUÇÃO",
    version="3.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware produção
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== SERVIÇO DE ARQUIVOS PWA ====================

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Home page not found")

@app.get("/{page_name}", include_in_schema=False)
async def serve_page(page_name: str):
    page_path = os.path.join(FRONTEND_PATH, page_name)
    if os.path.exists(page_path) and os.path.isfile(page_path):
        return FileResponse(page_path)
    
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    raise HTTPException(status_code=404, detail="Página não encontrada")

@app.get("/service-worker.js", include_in_schema=False)
async def serve_service_worker():
    sw_path = os.path.join(FRONTEND_PATH, "service-worker.js")
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service Worker not found")

@app.get("/manifest.json", include_in_schema=False)
async def serve_manifest():
    manifest_path = os.path.join(FRONTEND_PATH, "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="Manifest not found")

@app.get("/offline.html", include_in_schema=False)
async def serve_offline():
    offline_path = os.path.join(FRONTEND_PATH, "offline.html")
    if os.path.exists(offline_path):
        return FileResponse(offline_path)
    raise HTTPException(status_code=404, detail="Offline page not found")

@app.get("/icons/{icon_name}", include_in_schema=False)
async def serve_icon(icon_name: str):
    icon_path = os.path.join(FRONTEND_PATH, "icons", icon_name)
    if os.path.exists(icon_path):
        return FileResponse(icon_path)
    raise HTTPException(status_code=404, detail="Icon not found")

@app.get("/assets/{file_path:path}", include_in_schema=False)
async def serve_assets(file_path: str):
    asset_path = os.path.join(FRONTEND_PATH, file_path)
    if os.path.exists(asset_path):
        return FileResponse(asset_path)
    raise HTTPException(status_code=404, detail="Asset not found")

# Modelos Pydantic (mantidos)
class AuthRequest(BaseModel):
    token: str

class AccumulatorBuyRequest(BaseModel):
    amount: float
    symbol: str = "1HZ100V"
    growth_rate: float = 0.02
    duration: int = 60
    duration_unit: str = "t"
    
    @validator('amount')
    def validate_amount(cls, v):
        if v < 5 or v > 1000:
            raise ValueError('Amount must be between 5 and 1000')
        return v

class RobotConfig(BaseModel):
    strategy: str = "conservative"
    max_daily_loss: float = 100.0
    take_profit_ticks: int = 10
    stop_loss_ticks: int = 3
    trade_amount: float = 5.0
    growth_rate: float = 0.02

class ChatQuery(BaseModel):
    query: str

class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

# Autenticação (mantida)
def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    loginid_header = request.headers.get("X-LoginID")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token não fornecido")
    
    token = auth_header[7:]
    
    if not loginid_header:
        raise HTTPException(status_code=401, detail="LoginID não fornecido")
    
    if loginid_header not in active_tokens:
        raise HTTPException(status_code=401, detail="Usuário não autenticado")
    
    if active_tokens.get(loginid_header) != token:
        raise HTTPException(status_code=401, detail="Token inválido")
    
    session_key = f"session_{loginid_header}"
    if session_key in user_sessions:
        user_sessions[session_key]['last_activity'] = datetime.now().timestamp()
    else:
        user_sessions[session_key] = {
            'loginid': loginid_header,
            'created_at': datetime.now().timestamp(),
            'last_activity': datetime.now().timestamp()
        }
    
    return {
        "loginid": loginid_header,
        "token": token,
        "authenticated": True
    }

class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    async def is_rate_limited(self, key: str, limit: int, window: int = 60):
        now = datetime.now().timestamp()
        if key not in self.requests:
            self.requests[key] = []
        
        self.requests[key] = [req_time for req_time in self.requests[key] if now - req_time < window]
        
        if len(self.requests[key]) >= limit:
            return True
        
        self.requests[key].append(now)
        return False

rate_limiter = RateLimiter()

# ==================== ENDPOINTS PRODUÇÃO ====================

@app.get("/auth/login")
async def login_with_deriv():
    import urllib.parse
    
    state = secrets.token_urlsafe(16)
    
    params = urllib.parse.urlencode({
        "app_id": DERIV_APP_ID,
        "l": "pt",
        "brand": "deriv", 
        "redirect_uri": DERIV_REDIRECT_URL,
        "state": state
    })
    
    auth_url = f"https://oauth.deriv.com/oauth2/authorize?{params}"
    return RedirectResponse(auth_url)

@app.get("/auth/callback")
async def handle_oauth_callback(request: Request):
    try:
        query_params = dict(request.query_params)
        
        if "error" in query_params:
            error_msg = query_params.get("error", "Erro desconhecido")
            return RedirectResponse(url="/?auth_error=1", status_code=302)
        
        accounts = []
        i = 1
        while f"acct{i}" in query_params:
            loginid = query_params.get(f"acct{i}")
            token = query_params.get(f"token{i}")
            
            if loginid and token:
                account_info = {
                    "loginid": loginid,
                    "token": token,
                    "currency": query_params.get(f"cur{i}", "USD"),
                    "account_type": "demo" if loginid.startswith("VRTC") else "real"
                }
                accounts.append(account_info)
                
                active_tokens[loginid] = token
                session_key = f"session_{loginid}"
                user_sessions[session_key] = {
                    'loginid': loginid,
                    'created_at': datetime.now().timestamp(),
                    'last_activity': datetime.now().timestamp()
                }
            i += 1
        
        if not accounts:
            return RedirectResponse(url="/?auth_error=2", status_code=302)
        
        return RedirectResponse(url="/", status_code=302)
        
    except Exception as e:
        return RedirectResponse(url="/?auth_error=3", status_code=302)

@app.post("/api/auth/refresh")
async def refresh_session(request: Request):
    try:
        auth_header = request.headers.get("Authorization")
        loginid_header = request.headers.get("X-LoginID")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Token não fornecido")
        
        token = auth_header[7:]
        
        if loginid_header and token:
            active_tokens[loginid_header] = token
            session_key = f"session_{loginid_header}"
            user_sessions[session_key] = {
                'loginid': loginid_header,
                'created_at': datetime.now().timestamp(),
                'last_activity': datetime.now().timestamp()
            }
            
            return {
                "status": "success",
                "message": "Sessão restaurada",
                "loginid": loginid_header
            }
        else:
            raise HTTPException(status_code=400, detail="Dados de autenticação incompletos")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Falha ao restaurar sessão")

@app.post("/auth/logout")
async def logout_user(request: Request):
    try:
        auth_header = request.headers.get("Authorization")
        loginid_header = request.headers.get("X-LoginID")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Token não fornecido")
        
        token = auth_header[7:]
        
        if not loginid_header or loginid_header not in active_tokens:
            raise HTTPException(status_code=401, detail="Usuário não autenticado")
        
        if loginid_header in active_tokens:
            del active_tokens[loginid_header]
        session_key = f"session_{loginid_header}"
        if session_key in user_sessions:
            del user_sessions[session_key]
            
        return {"status": "success", "message": "Logout realizado"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no logout: {str(e)}")

@app.get("/api/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    return {
        "authenticated": True,
        "loginid": user['loginid'],
        "name": "Trader FinanceClick",
        "account_type": "demo" if user['loginid'].startswith("VRTC") else "real"
    }

# --- ENDPOINTS COM CONEXÃO ROBUSTA ---
@app.get("/api/balance")
async def get_account_balance(user: dict = Depends(get_current_user)):
    try:
        if deriv_service:
            real_balance = await deriv_service.get_balance(user['token'])
            if real_balance is not None:
                return {
                    "balance": {
                        "balance": real_balance,
                        "currency": "USD",
                        "loginid": user['loginid']
                    }
                }
        
        # Fallback para produção
        simulated_balance = 1000.00
        return {
            "balance": {
                "balance": simulated_balance,
                "currency": "USD", 
                "loginid": user['loginid']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Balance request error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get balance: {str(e)}")

@app.get("/api/symbols/accumulators")
async def get_accumulator_symbols():
    accumulator_symbols = [
        {"symbol": "1HZ10V", "display_name": "Volatility 10 Index"},
        {"symbol": "1HZ25V", "display_name": "Volatility 25 Index"},
        {"symbol": "1HZ50V", "display_name": "Volatility 50 Index"},
        {"symbol": "1HZ75V", "display_name": "Volatility 75 Index"},
        {"symbol": "1HZ100V", "display_name": "Volatility 100 Index"}
    ]
    return {"accumulator_symbols": accumulator_symbols}

@app.post("/api/accumulators/buy")
async def buy_accumulator_contract(
    buy_request: AccumulatorBuyRequest, 
    user: dict = Depends(get_current_user)
):
    try:
        if await rate_limiter.is_rate_limited(f"buy_{user['loginid']}", 10, 60):
            raise HTTPException(status_code=429, detail="Too many trade attempts")
        
        # Simulação de compra para produção
        import random
        contract_id = f"ACCU_{int(datetime.now().timestamp())}_{user['loginid']}"
        is_success = random.random() > 0.3
        profit_loss = buy_request.amount * buy_request.growth_rate * random.randint(5, 20) if is_success else -buy_request.amount
        
        return {
            "buy": {
                "contract_id": contract_id,
                "amount": buy_request.amount,
                "symbol": buy_request.symbol,
                "growth_rate": buy_request.growth_rate,
                "result": profit_loss,
                "status": "win" if is_success else "loss",
                "timestamp": datetime.now().isoformat(),
                "duration": buy_request.duration,
                "currency": "USD"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Accumulator buy error: {e}")
        raise HTTPException(status_code=500, detail=f"Accumulator buy failed: {str(e)}")

@app.post("/api/accumulators/proposal")
async def get_accumulator_proposal(buy_request: AccumulatorBuyRequest):
    import random
    potential_payout = buy_request.amount * (1 + buy_request.growth_rate * random.randint(8, 15))
    potential_return = potential_payout - buy_request.amount
    
    return {
        "proposal": {
            "display_value": f"{potential_payout:.2f}",
            "payout": potential_payout,
            "growth_rate": buy_request.growth_rate,
            "potential_return": potential_return,
            "return_percentage": (potential_return / buy_request.amount) * 100,
            "timestamp": datetime.now().isoformat()
        }
    }

# --- ENDPOINTS ESSENCIAIS PRODUÇÃO ---
@app.post("/api/contact")
async def submit_contact_form(contact_data: ContactRequest, request: Request):
    try:
        client_ip = request.client.host
        if await rate_limiter.is_rate_limited(f"contact_{client_ip}", 3, 300):
            raise HTTPException(status_code=429, detail="Muitas mensagens enviadas")
        
        contact_info = {
            **contact_data.dict(),
            "timestamp": datetime.now().isoformat(),
            "id": len(contact_messages) + 1,
        }
        
        contact_messages.append(contact_info)
        
        return {
            "status": "success",
            "message": "Mensagem enviada com sucesso!",
            "contact_id": contact_info["id"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar formulário: {str(e)}")

@app.post("/api/chatbot/ask")
async def chatbot_ask(query_data: ChatQuery, request: Request):
    client_ip = request.client.host
    if await rate_limiter.is_rate_limited(f"chatbot_{client_ip}", 20, 60):
        raise HTTPException(status_code=429, detail="Muitas requisições")
    
    query = query_data.query.lower()
    
    for regra in DEFAULT_KNOWLEDGE_BASE.get("regras", []):
        if any(keyword in query for keyword in regra.get("keywords", [])):
            return {"response": regra["resposta"]}
    
    return {
        "response": "Desculpe, sou especializado em Accumulator Options. Posso ajudar com: conexão Deriv, robô AI, estratégias, símbolos disponíveis, gestão de risco."
    }

# ✅ HEALTH CHECK PRODUÇÃO AVANÇADO
@app.get("/health")
async def production_health_check():
    """Health check específico para produção com verificação de conexão"""
    base_health = {
        "status": "healthy",
        "service": "FinanceClick AI Trading",
        "timestamp": datetime.now().isoformat(),
        "port": PORT,
        "environment": ENVIRONMENT,
        "version": "3.2.0",
        "branch": "master"
    }
    
    # Verificar saúde da conexão Deriv se o serviço estiver disponível
    if deriv_service:
        deriv_health = await deriv_service.production_health_check()
        base_health.update({
            "deriv_connection": deriv_health,
            "active_users": len(active_tokens),
            "user_sessions": len(user_sessions),
            "robot_active": robot_active
        })
    else:
        base_health.update({
            "deriv_connection": {"status": "service_not_ready"},
            "active_users": len(active_tokens),
            "user_sessions": len(user_sessions)
        })
    
    return base_health

@app.get("/api/health")
async def api_health_check():
    """Health check para API com status detalhado"""
    if deriv_service:
        deriv_status = await deriv_service.production_health_check()
    else:
        deriv_status = {"status": "service_not_initialized"}
    
    return {
        "status": "healthy",
        "deriv_connection": deriv_status,
        "active_users": len(active_tokens),
        "user_sessions": len(user_sessions),
        "environment": ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    }

# ✅ ENDPOINT DE STATUS DA CONEXÃO
@app.get("/api/connection/status")
async def connection_status():
    """Endpoint específico para verificar status da conexão Deriv"""
    if not deriv_service:
        return {"status": "service_not_initialized"}
    
    health = await deriv_service.production_health_check()
    return {
        "connection_status": health,
        "connection_attempts": deriv_service.connection_attempts,
        "max_retries": deriv_service.max_retries,
        "timestamp": datetime.now().isoformat()
    }

# ✅ BLOCO DE PRODUÇÃO - SEM RELOAD
if __name__ == "__main__":
    import uvicorn
    print(f"🏁 Iniciando servidor de PRODUÇÃO na porta {PORT}")
    uvicorn.run(
        app,  # ✅ Usar app diretamente para produção
        host="0.0.0.0", 
        port=PORT
        # ❌ SEM reload em produção
    )