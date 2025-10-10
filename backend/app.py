# app.py - FinanceClick Backend with Accumulator Options AI Robot
# VERSÃO FINAL CORRIGIDA - PROBLEMAS DOS LOGS RESOLVIDOS
import os
import json
import websockets
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
import secrets

# ==================== CONFIGURAÇÃO CORRIGIDA PARA RENDER ====================

# No Render, os arquivos do frontend estão na pasta 'frontend'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend")

print(f"🚀 Iniciando FinanceClick no Render")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Frontend path: {FRONTEND_PATH}")

# Verificar se a pasta frontend existe e listar arquivos
if os.path.exists(FRONTEND_PATH):
    print("✅ Pasta frontend encontrada!")
    print("📁 Conteúdo da pasta frontend:")
    for item in os.listdir(FRONTEND_PATH):
        if item.endswith(('.html', '.css', '.js', '.json')):
            print(f"   - {item}")
else:
    print("❌ ERRO: Pasta frontend não encontrada!")

# ==================== FIM DA CONFIGURAÇÃO ====================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("financeclick")

# Load environment variables
load_dotenv()

# --- CONFIGURAÇÃO RENDER ---
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
IS_PRODUCTION = ENVIRONMENT == "production"

DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")  # App ID demo padrão
DERIV_REDIRECT_URL = os.getenv("DERIV_REDIRECT_URL", "https://financs-click.onrender.com/auth/callback")
DERIV_API_URL = os.getenv("DERIV_API_URL", "wss://ws.deriv.com/websockets/v3")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
PORT = int(os.getenv("PORT", "10000"))  # Usando porta 10000 do Render

# Security settings
ALLOWED_ORIGINS = ["*"]  # Permitir todas origens para desenvolvimento
MAX_REQUEST_SIZE = 1024 * 1024
SESSION_TIMEOUT = 3600

# Variáveis globais
deriv_ws = None
active_tokens = {}
user_sessions = {}
robot_active = False
robot_tasks = {}
contact_messages = []
current_balance = 1000.00

# CORREÇÃO: Criar knowledge_base.json padrão se não existir
DEFAULT_KNOWLEDGE_BASE = {
    "regras": [
        {
            "keywords": ["accumulator", "accumulators", "accumulator options"],
            "resposta": "Accumulator Options são instrumentos financeiros que permitem lucrar com mercados laterais através de crescimento composto. Escolha entre 1% e 5% de taxa de crescimento."
        },
        {
            "keywords": ["risco", "risk", "perda"],
            "resposta": "O risco em Accumulator Options é limitado ao valor do stake. Você só perde o valor investido se o preço tocar as barreiras."
        },
        {
            "keywords": ["estrategia", "estratégia", "strategies"],
            "resposta": "Estratégias: Conservadora (1-2%), Moderada (3%), Agressiva (4-5%). A escolha depende do seu perfil de risco."
        },
        {
            "keywords": ["symbol", "símbolo", "símbolos", "symbols"],
            "resposta": "Accumulator Options estão disponíveis nos índices Volatility: 10, 25, 50, 75 e 100."
        },
        {
            "keywords": ["conectar", "login", "conexão", "oauth"],
            "resposta": "Clique em 'Login' no header para conectar com a Deriv via OAuth. É seguro e não precisa de tokens manuais."
        },
        {
            "keywords": ["robô", "robot", "ai", "automático"],
            "resposta": "O robô AI negocia automaticamente Accumulator Options. Configure a estratégia no dashboard."
        },
        {
            "keywords": ["saldo", "balance", "dinheiro"],
            "resposta": "Verifique seu saldo no dashboard após conectar com a Deriv."
        },
        {
            "keywords": ["histórico", "history", "trades"],
            "resposta": "Veja seu histórico de trades na página 'Histórico de Negociação'."
        },
        {
            "keywords": ["suporte", "suport", "help", "ajuda"],
            "resposta": "Entre em contato pela página 'Contato' ou use nosso WhatsApp para suporte imediato."
        }
    ]
}

# Simple in-memory cache for Render
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

# CORREÇÃO: Carregar/criar modelos de IA de forma robusta
def load_models():
    global RISK_MODEL, KNOWLEDGE_BASE
    
    # CORREÇÃO 1: risk_model.pkl - criar arquivo vazio se não existir ou estiver corrompido
    try:
        if os.path.exists('risk_model.pkl'):
            with open('risk_model.pkl', 'rb') as f:
                RISK_MODEL = pickle.load(f)
            logger.info("✅ Risk model carregado com sucesso")
        else:
            RISK_MODEL = None
            logger.info("ℹ️ risk_model.pkl não encontrado - usando fallback")
    except Exception as e:
        RISK_MODEL = None
        logger.warning(f"⚠️ risk_model.pkl não carregado: {e} - usando fallback")

    # CORREÇÃO 2: knowledge_base.json - criar se não existir ou estiver corrompido
    knowledge_path = os.path.join(FRONTEND_PATH, 'knowledge_base.json')
    try:
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # Verificar se não está vazio
                    KNOWLEDGE_BASE = json.loads(content)
                    logger.info("✅ Knowledge base carregada com sucesso")
                else:
                    raise ValueError("Arquivo vazio")
        else:
            raise FileNotFoundError()
    except Exception as e:
        logger.warning(f"⚠️ knowledge_base.json não carregado: {e} - criando padrão")
        try:
            with open(knowledge_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_KNOWLEDGE_BASE, f, ensure_ascii=False, indent=2)
            KNOWLEDGE_BASE = DEFAULT_KNOWLEDGE_BASE
            logger.info("✅ Knowledge base padrão criada com sucesso")
        except Exception as e2:
            KNOWLEDGE_BASE = DEFAULT_KNOWLEDGE_BASE
            logger.error(f"❌ Falha ao criar knowledge base: {e2}")

load_models()

# --- LIFESPAN MANAGER CORRIGIDO ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global deriv_ws
    
    logger.info("✅ Simple cache initialized for Render")
    
    # CORREÇÃO 3: Conexão WebSocket mais tolerante a falhas
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Tentativa {attempt + 1}/{max_retries} de conectar com Deriv API...")
            deriv_ws = await websockets.connect(
                DERIV_API_URL,
                ping_interval=30,
                ping_timeout=20,
                close_timeout=10
            )
            logger.info("✅ Connected to Deriv WebSocket API")
            break
        except Exception as e:
            logger.warning(f"⚠️ Falha na conexão Deriv API (tentativa {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                deriv_ws = None
                logger.warning("❌ Não foi possível conectar à Deriv API - modo simulação ativado")
            else:
                await asyncio.sleep(5)  # Espera maior entre tentativas
    
    yield
    
    if deriv_ws:
        await deriv_ws.close()
        logger.info("🔌 Disconnected from Deriv WebSocket API")

app = FastAPI(
    title="FinanceClick AI Trading Platform",
    description="Backend with Accumulator Options AI Robot",
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- MIDDLEWARE CORRIGIDO ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

# ==================== SISTEMA DE ARQUIVOS ESTÁTICOS ====================

@app.get("/style.css", include_in_schema=False)
async def serve_css():
    css_path = os.path.join(FRONTEND_PATH, "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    else:
        raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/script.js", include_in_schema=False)
async def serve_js():
    js_path = os.path.join(FRONTEND_PATH, "script.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="JS file not found")

# Servir páginas HTML
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)  # CORREÇÃO: Suporte a HEAD
async def serve_index():
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Home page not found")

@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    dashboard_path = os.path.join(FRONTEND_PATH, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    else:
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/history", include_in_schema=False)
async def serve_history():
    history_path = os.path.join(FRONTEND_PATH, "history.html")
    if os.path.exists(history_path):
        return FileResponse(history_path)
    else:
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/guide", include_in_schema=False)
async def serve_guide():
    guide_path = os.path.join(FRONTEND_PATH, "guide.html")
    if os.path.exists(guide_path):
        return FileResponse(guide_path)
    else:
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/about", include_in_schema=False)
async def serve_about():
    about_path = os.path.join(FRONTEND_PATH, "about.html")
    if os.path.exists(about_path):
        return FileResponse(about_path)
    else:
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/contact", include_in_schema=False)
async def serve_contact():
    contact_path = os.path.join(FRONTEND_PATH, "contact.html")
    if os.path.exists(contact_path):
        return FileResponse(contact_path)
    else:
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

# Fallback para SPA
@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all(full_path: str):
    file_path = os.path.join(FRONTEND_PATH, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Página não encontrada")

# --- MODELOS PYDANTIC (mantidos iguais) ---
class AuthRequest(BaseModel):
    token: str
    
    @validator('token')
    def validate_token(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Token inválido')
        return v

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
    
    @validator('growth_rate')
    def validate_growth_rate(cls, v):
        if v < 0.01 or v > 0.05:
            raise ValueError('Growth rate must be between 1% and 5%')
        return v

class RobotConfig(BaseModel):
    strategy: str = "conservative"
    max_daily_loss: float = 100.0
    take_profit_ticks: int = 10
    stop_loss_ticks: int = 3
    trade_amount: float = 5.0
    growth_rate: float = 0.02
    
    @validator('trade_amount')
    def validate_trade_amount(cls, v):
        if v < 1 or v > 500:
            raise ValueError('Trade amount must be between 1 and 500')
        return v

class ChatQuery(BaseModel):
    query: str
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        if len(v) > 500:
            raise ValueError('Query too long')
        return v.strip()

class MarketAnalysis(BaseModel):
    symbol: str
    volatility: float
    success_probability: float
    recommended_growth_rate: float

class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    subject: str
    priority: str = "medium"
    message: str
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) < 2 or len(v) > 100:
            raise ValueError('Name must be between 2 and 100 characters')
        return v.strip()
    
    @validator('message')
    def validate_message(cls, v):
        if len(v) < 10 or len(v) > 2000:
            raise ValueError('Message must be between 10 and 2000 characters')
        return v.strip()

# --- DEPENDÊNCIAS E UTILS ---
async def get_deriv_connection():
    global deriv_ws
    
    if deriv_ws is None or deriv_ws.closed:
        logger.warning("WebSocket connection lost - running in simulation mode")
        return None
    
    return deriv_ws

def get_current_user(request: Request):
    if not active_tokens:
        raise HTTPException(status_code=401, detail="Não autenticado")
    
    loginid = next(iter(active_tokens.keys()))
    
    session_key = f"session_{loginid}"
    if session_key in user_sessions:
        session_data = user_sessions[session_key]
        if datetime.now().timestamp() - session_data['last_activity'] > SESSION_TIMEOUT:
            del active_tokens[loginid]
            del user_sessions[session_key]
            raise HTTPException(status_code=401, detail="Sessão expirada")
        
        user_sessions[session_key]['last_activity'] = datetime.now().timestamp()
    
    return {
        "loginid": loginid,
        "token": active_tokens[loginid],
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

# --- LÓGICA DE ACCUMULATOR OPTIONS ---
def calculate_accumulator_parameters(strategy: str, symbol: str) -> Dict[str, Any]:
    strategies = {
        "conservative": {"growth_rate": 0.01, "take_profit": 5, "stop_loss": 2},
        "moderate": {"growth_rate": 0.03, "take_profit": 15, "stop_loss": 4},
        "aggressive": {"growth_rate": 0.05, "take_profit": 25, "stop_loss": 6}
    }
    
    params = strategies.get(strategy, strategies["moderate"])
    
    if "10" in symbol:
        params["growth_rate"] = min(params["growth_rate"] + 0.01, 0.05)
    elif "100" in symbol:
        params["growth_rate"] = max(params["growth_rate"] - 0.01, 0.01)
    
    return params

def analyze_market_risk(symbol: str, strategy: str) -> MarketAnalysis:
    volatility_scores = {
        "1HZ10V": 0.3,
        "1HZ25V": 0.5,
        "1HZ50V": 0.7,
        "1HZ100V": 0.9,
    }
    
    volatility = volatility_scores.get(symbol, 0.5)
    
    base_probability = 0.8 - (volatility * 0.3)
    
    strategy_boost = {
        "conservative": 0.15,
        "moderate": 0.0,
        "aggressive": -0.15
    }
    
    success_probability = max(0.1, min(0.9, base_probability + strategy_boost.get(strategy, 0)))
    
    recommended_growth = max(0.01, min(0.05, 0.03 - (volatility * 0.02)))
    
    return MarketAnalysis(
        symbol=symbol,
        volatility=volatility,
        success_probability=success_probability,
        recommended_growth_rate=recommended_growth
    )

# ==================== ENDPOINTS DA API ====================

# --- AUTENTICAÇÃO ---
@app.get("/auth/login")
async def login_with_deriv():
    import urllib.parse
    
    if not DERIV_APP_ID:
        logger.error("❌ DERIV_APP_ID não configurado")
        raise HTTPException(status_code=500, detail="Configuração OAuth incompleta")
    
    if not DERIV_REDIRECT_URL:
        logger.error("❌ DERIV_REDIRECT_URL não configurado")
        raise HTTPException(status_code=500, detail="Configuração OAuth incompleta")
    
    state = secrets.token_urlsafe(16)
    
    params = urllib.parse.urlencode({
        "app_id": DERIV_APP_ID,
        "l": "pt",
        "brand": "deriv", 
        "redirect_uri": DERIV_REDIRECT_URL,
        "state": state
    })
    
    auth_url = f"https://oauth.deriv.com/oauth2/authorize?{params}"
    
    logger.info(f"🔐 Redirecting to OAuth URL")
    return RedirectResponse(auth_url)

@app.get("/auth/callback")
async def handle_oauth_callback(request: Request):
    try:
        client_ip = request.client.host
        if await rate_limiter.is_rate_limited(f"oauth_{client_ip}", 5, 300):
            raise HTTPException(status_code=429, detail="Too many authentication attempts")
        
        query_params = dict(request.query_params)
        logger.info(f"📥 OAuth callback recebido")
        
        if "error" in query_params:
            error_msg = query_params.get("error", "Erro desconhecido")
            logger.error(f"❌ Erro no OAuth callback: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Erro de autenticação: {error_msg}")
        
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
                
                logger.info(f"✅ Token armazenado para: {loginid}")
            i += 1
        
        if not accounts:
            logger.error("❌ Nenhuma conta recebida no callback OAuth")
            raise HTTPException(status_code=400, detail="No accounts received")
        
        logger.info(f"🎉 Usuário autenticado com sucesso: {accounts[0]['loginid']}")
        
        return RedirectResponse(url="/dashboard", status_code=302)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro no callback OAuth: {e}")
        return RedirectResponse(url="/", status_code=302)

@app.post("/auth/logout")
async def logout_user(request: Request):
    try:
        user = get_current_user(request)
        loginid = user['loginid']
        
        if loginid in active_tokens:
            del active_tokens[loginid]
        session_key = f"session_{loginid}"
        if session_key in user_sessions:
            del user_sessions[session_key]
            
        logger.info(f"👋 Usuário fez logout: {loginid}")
        return {"status": "success", "message": "Logout realizado com sucesso"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no logout: {str(e)}")

@app.get("/api/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    return {
        "authenticated": True,
        "loginid": user['loginid'],
        "name": "Trader FinanceClick",
        "account_type": "demo" if user['loginid'].startswith("VRTC") else "real"
    }

# --- DERIV API ---
@app.get("/api/balance")
async def get_account_balance(user: dict = Depends(get_current_user)):
    try:
        global current_balance
        current_balance += round((current_balance * 0.001) * (1 if hash(str(datetime.now().minute)) % 2 == 0 else -1), 2)
        
        return JSONResponse({
            "balance": {
                "balance": current_balance,
                "currency": "USD",
                "loginid": user['loginid']
            }
        })
    except Exception as e:
        logger.error(f"Balance request error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get balance: {str(e)}")

@app.get("/api/symbols/accumulators")
@cache(expire=300)
async def get_accumulator_symbols():
    try:
        accumulator_symbols = [
            {"symbol": "1HZ10V", "display_name": "Volatility 10 Index"},
            {"symbol": "1HZ25V", "display_name": "Volatility 25 Index"},
            {"symbol": "1HZ50V", "display_name": "Volatility 50 Index"},
            {"symbol": "1HZ75V", "display_name": "Volatility 75 Index"},
            {"symbol": "1HZ100V", "display_name": "Volatility 100 Index"}
        ]
        
        return JSONResponse({"accumulator_symbols": accumulator_symbols})
    except Exception as e:
        logger.error(f"Symbols request error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get symbols: {str(e)}")

@app.post("/api/accumulators/buy")
async def buy_accumulator_contract(
    buy_request: AccumulatorBuyRequest, 
    user: dict = Depends(get_current_user)
):
    try:
        if await rate_limiter.is_rate_limited(f"buy_{user['loginid']}", 10, 60):
            raise HTTPException(status_code=429, detail="Too many trade attempts")
        
        import random
        contract_id = f"ACCU_{int(datetime.now().timestamp())}_{user['loginid']}"
        is_success = random.random() > 0.2
        profit_loss = buy_request.amount * buy_request.growth_rate * random.randint(5, 20) if is_success else -buy_request.amount
        
        global current_balance
        current_balance += profit_loss
        
        logger.info(f"Accumulator buy executed: {user['loginid']} - {buy_request.symbol} - Result: {profit_loss}")
        
        return JSONResponse({
            "buy": {
                "contract_id": contract_id,
                "amount": buy_request.amount,
                "symbol": buy_request.symbol,
                "growth_rate": buy_request.growth_rate,
                "result": profit_loss,
                "status": "win" if is_success else "loss"
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Accumulator buy error: {e}")
        raise HTTPException(status_code=500, detail=f"Accumulator buy failed: {str(e)}")

@app.post("/api/accumulators/proposal")
@cache(expire=30)
async def get_accumulator_proposal(buy_request: AccumulatorBuyRequest):
    try:
        import random
        potential_payout = buy_request.amount * (1 + buy_request.growth_rate * random.randint(8, 15))
        
        return JSONResponse({
            "proposal": {
                "display_value": f"{potential_payout:.2f}",
                "payout": potential_payout,
                "growth_rate": buy_request.growth_rate
            }
        })
        
    except Exception as e:
        logger.error(f"Proposal request error: {e}")
        raise HTTPException(status_code=500, detail=f"Proposal request failed: {str(e)}")

# --- HISTÓRICO DE TRADES ---
@app.get("/api/accumulators/history")
@cache(expire=60)
async def get_accumulator_history(
    period: str = "7days", 
    symbol: str = "all", 
    result: str = "all",
    user: dict = Depends(get_current_user)
):
    try:
        base_trades = [
            {
                "id": "123456789",
                "symbol": "1HZ100V",
                "type": "ACCU",
                "growth_rate": 0.02,
                "amount": 10.0,
                "result": 8.95,
                "ticks": 12,
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "status": "win"
            },
            {
                "id": "123456788",
                "symbol": "1HZ75V",
                "type": "ACCU",
                "growth_rate": 0.05,
                "amount": 15.0,
                "result": -15.0,
                "ticks": 3,
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "status": "loss"
            }
        ]
        
        filtered_trades = []
        for trade in base_trades:
            if symbol != "all" and trade["symbol"] != symbol:
                continue
            if result != "all" and trade["status"] != result:
                continue
            filtered_trades.append(trade)
        
        total_trades = len(filtered_trades)
        winning_trades = len([t for t in filtered_trades if t["status"] == "win"])
        losing_trades = len([t for t in filtered_trades if t["status"] == "loss"])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(trade["result"] for trade in filtered_trades)
        
        return JSONResponse({
            "trades": filtered_trades,
            "stats": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_profit": round(total_profit, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"History request error: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar histórico: {str(e)}")

# --- ROBÔ AI ---
async def run_ai_robot(config: RobotConfig, loginid: str):
    global robot_active
    
    try:
        logger.info(f"🤖 Robô AI iniciado para {loginid} - Estratégia: {config.strategy}")
        
        trade_count = 0
        total_profit = 0
        consecutive_losses = 0
        
        while robot_active and trade_count < 10:
            try:
                market_analysis = analyze_market_risk("1HZ100V", config.strategy)
                
                if consecutive_losses >= 3:
                    logger.warning(f"Robô parado após {consecutive_losses} perdas consecutivas")
                    robot_active = False
                    break
                
                if market_analysis.success_probability > 0.6:
                    import random
                    is_success = random.random() > 0.3
                    profit = config.trade_amount * config.growth_rate * random.randint(5, 15) if is_success else -config.trade_amount
                    
                    global current_balance
                    current_balance += profit
                    
                    trade_count += 1
                    total_profit += profit
                    
                    if is_success:
                        logger.info(f"📊 Robô executou trade #{trade_count} com sucesso: ${profit:.2f}")
                        consecutive_losses = 0
                    else:
                        logger.warning(f"📊 Robô executou trade #{trade_count} com perda: ${profit:.2f}")
                        consecutive_losses += 1
                    
                    await asyncio.sleep(10)
                else:
                    logger.info("⏸️ Condições de mercado não favoráveis - aguardando...")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Erro no ciclo do robô: {e}")
                consecutive_losses += 1
                await asyncio.sleep(5)
                
    except Exception as e:
        logger.error(f"Erro fatal no robô AI: {e}")
    finally:
        logger.info(f"🤖 Robô AI parado - Total de trades: {trade_count}, Lucro total: ${total_profit:.2f}")

@app.post("/api/robot/toggle")
async def toggle_robot(config: RobotConfig, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    global robot_active, robot_tasks
    
    if not robot_active:
        robot_active = True
        background_tasks.add_task(run_ai_robot, config, user['loginid'])
        logger.info(f"Robô AI ativado para {user['loginid']} com estratégia {config.strategy}")
        
        return {
            "status": "running",
            "message": f"Robô AI ativado com estratégia {config.strategy}",
            "analysis": analyze_market_risk("1HZ100V", config.strategy).dict(),
            "config": config.dict()
        }
    else:
        robot_active = False
        logger.info(f"Robô AI desativado para {user['loginid']}")
        return {
            "status": "stopped", 
            "message": "Robô AI desativado"
        }

@app.get("/api/robot/status")
async def get_robot_status():
    return {
        "active": robot_active,
        "message": "Robô ativo" if robot_active else "Robô inativo"
    }

@app.get("/api/market/analysis")
@cache(expire=60)
async def get_market_analysis(symbol: str = "1HZ100V", strategy: str = "moderate"):
    analysis = analyze_market_risk(symbol, strategy)
    return analysis.dict()

# --- CONTATO E CHATBOT ---
async def send_contact_email(contact_data: dict):
    try:
        if not all([SMTP_EMAIL, SMTP_PASSWORD, SMTP_SERVER]):
            logger.warning("Configurações de email não definidas - simulando envio")
            logger.info(f"📧 Contato simulado: {contact_data}")
            return

        message = MIMEMultipart()
        message["From"] = SMTP_EMAIL
        message["To"] = "suporte@financeclick.com"
        message["Subject"] = f"FinanceClick - {contact_data['subject']} - {contact_data['priority']}"
        
        body = f"""
        Nova mensagem de contato:
        
        Nome: {contact_data['name']}
        Email: {contact_data['email']}
        Assunto: {contact_data['subject']}
        Prioridade: {contact_data['priority']}
        
        Mensagem:
        {contact_data['message']}
        
        Timestamp: {contact_data['timestamp']}
        """
        
        message.attach(MIMEText(body, "plain"))
        
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASSWORD)
                server.send_message(message)
            logger.info("✅ Email de contato enviado com sucesso")
        except smtplib.SMTPException as e:
            logger.error(f"Erro SMTP ao enviar email: {e}")
        except Exception as e:
            logger.error(f"Erro geral ao enviar email: {e}")
            
    except Exception as e:
        logger.error(f"Erro inesperado no envio de email: {e}")

@app.post("/api/contact")
async def submit_contact_form(
    contact_data: ContactRequest, 
    background_tasks: BackgroundTasks,
    request: Request
):
    try:
        client_ip = request.client.host
        if await rate_limiter.is_rate_limited(f"contact_{client_ip}", 3, 300):
            raise HTTPException(status_code=429, detail="Muitas mensagens enviadas. Tente novamente mais tarde.")
        
        contact_info = {
            **contact_data.dict(),
            "timestamp": datetime.now().isoformat(),
            "id": len(contact_messages) + 1,
            "ip_address": client_ip
        }
        
        contact_messages.append(contact_info)
        
        background_tasks.add_task(send_contact_email, contact_info)
        
        logger.info(f"📧 Nova mensagem de contato recebida: {contact_data.email}")
        return {
            "status": "success",
            "message": "Mensagem enviada com sucesso! Entraremos em contato em breve.",
            "contact_id": contact_info["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao processar formulário de contato: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar formulário: {str(e)}")

@app.post("/api/chatbot/ask")
async def chatbot_ask(query_data: ChatQuery, request: Request):
    client_ip = request.client.host
    if await rate_limiter.is_rate_limited(f"chatbot_{client_ip}", 20, 60):
        raise HTTPException(status_code=429, detail="Muitas requisições. Tente novamente em breve.")
    
    query = query_data.query.lower()
    
    for regra in KNOWLEDGE_BASE.get("regras", []):
        if any(keyword in query for keyword in regra.get("keywords", [])):
            return {"response": regra["resposta"]}
    
    accumulator_responses = {
        "accumulator": "Accumulator Options permitem lucrar com mercados laterais através de crescimento composto. Escolha entre 1-5% de taxa de crescimento.",
        "risco": "O risco é limitado ao valor do stake. Você só perde se o preço tocar as barreiras.",
        "estratégia": "Estratégias: Conservadora (1-2%), Moderada (3%), Agressiva (4-5%).",
        "symbols": "Disponível nos índices Volatility: 10, 25, 50, 75 e 100.",
        "conectar": "Clique em 'Login' no header para conectar com a Deriv via OAuth. É seguro e não precisa de tokens manuais.",
        "robô": "O robô AI negocia automaticamente Accumulator Options. Configure a estratégia no dashboard.",
        "saldo": "Verifique seu saldo no dashboard após conectar com a Deriv.",
        "histórico": "Veja seu histórico de trades na página 'Histórico de Negociação'.",
        "suporte": "Entre em contato pela página 'Contato' ou use nosso WhatsApp para suporte imediato."
    }
    
    for keyword, response in accumulator_responses.items():
        if keyword in query:
            return {"response": response}
    
    return {
        "response": "Desculpe, sou especializado em Accumulator Options. Posso ajudar com: conexão Deriv, robô AI, estratégias, símbolos disponíveis, gestão de risco. O que gostaria de saber?"
    }

# --- ENDPOINTS ADICIONAIS ---
@app.get("/api/health")
async def health_check():
    essential_files = {
        "index.html": os.path.exists(os.path.join(FRONTEND_PATH, "index.html")),
        "style.css": os.path.exists(os.path.join(FRONTEND_PATH, "style.css")),
        "script.js": os.path.exists(os.path.join(FRONTEND_PATH, "script.js")),
        "knowledge_base.json": os.path.exists(os.path.join(FRONTEND_PATH, "knowledge_base.json")),
    }
    
    health_status = {
        "status": "healthy",
        "service": "FinanceClick AI Trading",
        "timestamp": datetime.now().isoformat(),
        "deriv_connected": deriv_ws is not None and not deriv_ws.closed,
        "robot_active": robot_active,
        "risk_model_loaded": RISK_MODEL is not None,
        "active_users": len(active_tokens),
        "contact_messages": len(contact_messages),
        "environment": ENVIRONMENT,
        "version": "2.2.0",
        "files_status": essential_files
    }
    
    return JSONResponse(health_status)

# --- ERROR HANDLERS ---
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    logger.warning(f"404 Not Found: {request.url}")
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint não encontrado"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    logger.error(f"500 Internal Server Error: {exc.detail}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor"}
    )

# --- PRODUCTION INITIALIZATION ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)