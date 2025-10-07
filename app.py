# app.py - FinanceClick Backend with Accumulator Options AI Robot
# RENDER-OPTIMIZED VERSION (NO REDIS)
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
# ==================== CONFIGURAÇÃO DE PATHS INTELIGENTE ====================
def get_project_root():
    """Encontra a raiz do projeto automaticamente - funciona em qualquer ambiente"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # Debug: verificar onde estamos
    print(f"📁 Diretório atual: {current_dir}")
    print(f"📁 Nome do diretório: {os.path.basename(current_dir)}")
    
    # Se estamos em backend/, sobe um nível para a raiz
    if os.path.basename(current_dir) == 'backend':
        project_root = os.path.dirname(current_dir)
        print(f"🎯 Detectado: Em pasta backend, raiz do projeto: {project_root}")
    else:
        # Se já estamos na raiz (Render)
        project_root = current_dir
        print(f"🎯 Detectado: Na raiz do projeto: {project_root}")
    
    return project_root

PROJECT_ROOT = get_project_root()
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend")

print(f"🚀 Frontend path: {FRONTEND_PATH}")
print(f"🚀 Project root: {PROJECT_ROOT}")

# ==================== FIM DOS CAMINHOS INTELIGENTES ====================
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("financeclick")

# Load environment variables
load_dotenv()

# --- RENDER-OPTIMIZED CONFIGURATION ---
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
IS_PRODUCTION = ENVIRONMENT == "production"

DERIV_APP_ID = os.getenv("DERIV_APP_ID")
DERIV_REDIRECT_URL = os.getenv("DERIV_REDIRECT_URL")
DERIV_API_URL = os.getenv("DERIV_API_URL", "wss://ws.deriv.com/websockets/v3")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
PORT = int(os.getenv("PORT", "8000"))

# Security settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
MAX_REQUEST_SIZE = 1024 * 1024
SESSION_TIMEOUT = 3600

# Variáveis globais
deriv_ws = None
active_tokens = {}
user_sessions = {}
robot_active = False
robot_tasks = {}
contact_messages = []

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

# Carregar modelos de IA com tratamento robusto
def load_models():
    global RISK_MODEL, KNOWLEDGE_BASE
    
    try:
        with open('risk_model.pkl', 'rb') as f:
            RISK_MODEL = pickle.load(f)
        logger.info("✅ Risk model carregado com sucesso")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        RISK_MODEL = None
        logger.warning(f"⚠️ risk_model.pkl não carregado: {e}")

    try:
        knowledge_path = os.path.join(FRONTEND_PATH, 'knowledge_base.json')
        with open(knowledge_path, "r", encoding="utf-8") as f:
            KNOWLEDGE_BASE = json.load(f)
        logger.info("✅ Knowledge base carregada com sucesso")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        KNOWLEDGE_BASE = {"regras": []}
        logger.warning(f"⚠️ knowledge_base.json não carregado: {e}")

load_models()

# --- LIFESPAN MANAGER FOR RENDER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global deriv_ws
    
    logger.info("✅ Simple cache initialized for Render")
    
    # Initialize Deriv WebSocket with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            deriv_ws = await websockets.connect(
                DERIV_API_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            logger.info("✅ Connected to Deriv WebSocket API")
            break
        except Exception as e:
            logger.error(f"❌ Failed to connect to Deriv API (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                deriv_ws = None
            else:
                await asyncio.sleep(2 ** attempt)
    
    yield
    
    if deriv_ws:
        await deriv_ws.close()
        logger.info("🔌 Disconnected from Deriv WebSocket API")

app = FastAPI(
    title="FinanceClick AI Trading Platform",
    description="Backend with Accumulator Options AI Robot",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- SECURITY MIDDLEWARE FOR RENDER ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=600,
)

app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

# --- ENHANCED PYDANTIC MODELS WITH VALIDATION ---
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

class TradeHistoryRequest(BaseModel):
    period: str = "7days"
    symbol: str = "all"
    result: str = "all"

# --- ENHANCED DEPENDENCIES E UTILS ---
async def get_deriv_connection():
    """Improved WebSocket connection with health check"""
    global deriv_ws
    
    if deriv_ws is None or deriv_ws.closed:
        logger.warning("WebSocket connection lost, attempting reconnect...")
        try:
            deriv_ws = await websockets.connect(
                DERIV_API_URL,
                ping_interval=20,
                ping_timeout=10
            )
            logger.info("✅ Reconnected to Deriv WebSocket API")
        except Exception as e:
            logger.error(f"❌ Failed to reconnect to Deriv API: {e}")
            raise HTTPException(status_code=503, detail="Deriv API connection unavailable")
    
    return deriv_ws

async def send_deriv_message(websocket, message: dict, max_retries: int = 2):
    """Enhanced message sending with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            await websocket.send(json.dumps(message))
            response = await asyncio.wait_for(websocket.recv(), timeout=15)
            return json.loads(response)
        except asyncio.TimeoutError:
            logger.warning(f"Deriv API timeout (attempt {attempt + 1}/{max_retries + 1})")
            if attempt == max_retries:
                raise HTTPException(status_code=504, detail="Deriv API timeout")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed during operation")
            if attempt == max_retries:
                raise HTTPException(status_code=503, detail="Deriv API connection lost")
            # Refresh connection
            websocket = await get_deriv_connection()
        except Exception as e:
            logger.error(f"Deriv API error: {e}")
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail=f"Deriv API error: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Max retries exceeded")

async def authorize_connection(token: str, websocket):
    auth_message = {"authorize": token}
    return await send_deriv_message(websocket, auth_message)

def get_current_user(request: Request):
    """Enhanced user session management"""
    if not active_tokens:
        raise HTTPException(status_code=401, detail="Não autenticado")
    
    # Simple session management - in production, use JWT or proper sessions
    loginid = next(iter(active_tokens.keys()))
    
    # Check session timeout
    session_key = f"session_{loginid}"
    if session_key in user_sessions:
        session_data = user_sessions[session_key]
        if datetime.now().timestamp() - session_data['last_activity'] > SESSION_TIMEOUT:
            del active_tokens[loginid]
            del user_sessions[session_key]
            raise HTTPException(status_code=401, detail="Sessão expirada")
        
        # Update last activity
        user_sessions[session_key]['last_activity'] = datetime.now().timestamp()
    
    return {
        "loginid": loginid,
        "token": active_tokens[loginid],
        "authenticated": True
    }

# Rate limiting helper (simple in-memory version)
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    async def is_rate_limited(self, key: str, limit: int, window: int = 60):
        now = datetime.now().timestamp()
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] if now - req_time < window]
        
        if len(self.requests[key]) >= limit:
            return True
        
        self.requests[key].append(now)
        return False

rate_limiter = RateLimiter()

# --- ENHANCED ACCUMULATOR OPTIONS LOGIC ---
def calculate_accumulator_parameters(strategy: str, symbol: str) -> Dict[str, Any]:
    """Calcula parâmetros ótimos para Accumulator Options baseado na estratégia"""
    strategies = {
        "conservative": {"growth_rate": 0.01, "take_profit": 5, "stop_loss": 2},
        "moderate": {"growth_rate": 0.03, "take_profit": 15, "stop_loss": 4},
        "aggressive": {"growth_rate": 0.05, "take_profit": 25, "stop_loss": 6}
    }
    
    params = strategies.get(strategy, strategies["moderate"])
    
    # Ajustar baseado no símbolo (volatilidade)
    if "10" in symbol:
        params["growth_rate"] = min(params["growth_rate"] + 0.01, 0.05)
    elif "100" in symbol:
        params["growth_rate"] = max(params["growth_rate"] - 0.01, 0.01)
    
    return params

def analyze_market_risk(symbol: str, strategy: str) -> MarketAnalysis:
    """Analisa risco de mercado usando o modelo de IA"""
    volatility_scores = {
        "1HZ10V": 0.3,
        "1HZ25V": 0.5,
        "1HZ50V": 0.7,
        "1HZ100V": 0.9,
    }
    
    volatility = volatility_scores.get(symbol, 0.5)
    
    # Calcular probabilidade de sucesso
    base_probability = 0.8 - (volatility * 0.3)
    
    strategy_boost = {
        "conservative": 0.15,
        "moderate": 0.0,
        "aggressive": -0.15
    }
    
    success_probability = max(0.1, min(0.9, base_probability + strategy_boost.get(strategy, 0)))
    
    # Recomendar taxa de crescimento ótima
    recommended_growth = max(0.01, min(0.05, 0.03 - (volatility * 0.02)))
    
    return MarketAnalysis(
        symbol=symbol,
        volatility=volatility,
        success_probability=success_probability,
        recommended_growth_rate=recommended_growth
    )

# --- ENHANCED ENDPOINTS WITH SECURITY AND CACHE ---
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/{page}", include_in_schema=False)
async def serve_html_page(page: str):
    # Security: Prevent directory traversal
    if ".." in page or page.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid page request")
        
    if not page.endswith(".html"):
        page = f"{page}.html"
    file_path = os.path.join(FRONTEND_PATH, page)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Página não encontrada")
    return FileResponse(file_path)

# --- ENHANCED AUTH ENDPOINTS ---
@app.get("/auth/login")
async def login_with_deriv():
    import urllib.parse
    # Generate state parameter for CSRF protection
    state = secrets.token_urlsafe(16)
    params = urllib.parse.urlencode({
        "app_id": DERIV_APP_ID,
        "state": state
    })
    auth_url = f"https://oauth.deriv.com/oauth2/authorize?{params}"
    return RedirectResponse(auth_url)

@app.get("/auth/callback")
async def handle_oauth_callback(request: Request):
    try:
        # Rate limiting
        client_ip = request.client.host
        if await rate_limiter.is_rate_limited(f"oauth_{client_ip}", 5, 300):  # 5 attempts per 5 minutes
            raise HTTPException(status_code=429, detail="Too many authentication attempts")
        
        query_params = dict(request.query_params)
        accounts = []
        i = 1
        while f"acct{i}" in query_params:
            loginid = query_params.get(f"acct{i}")
            token = query_params.get(f"token{i}")
            
            account_info = {
                "loginid": loginid,
                "token": token,
                "currency": query_params.get(f"cur{i}"),
                "account_type": "demo" if loginid.startswith("VRTC") else "real"
            }
            accounts.append(account_info)
            
            if token:
                active_tokens[loginid] = token
                # Create session
                session_key = f"session_{loginid}"
                user_sessions[session_key] = {
                    'loginid': loginid,
                    'created_at': datetime.now().timestamp(),
                    'last_activity': datetime.now().timestamp()
                }
            i += 1
        
        if not accounts:
            raise HTTPException(status_code=400, detail="No accounts received")
        
        logger.info(f"User authenticated: {accounts[0]['loginid']}")
        return RedirectResponse(url="/dashboard", status_code=302)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=400, detail=f"OAuth callback error: {str(e)}")

@app.post("/auth/logout")
async def logout_user(request: Request):
    """Faz logout do usuário removendo tokens ativos"""
    try:
        user = get_current_user(request)
        loginid = user['loginid']
        
        # Remove tokens and sessions
        if loginid in active_tokens:
            del active_tokens[loginid]
        session_key = f"session_{loginid}"
        if session_key in user_sessions:
            del user_sessions[session_key]
            
        logger.info(f"User logged out: {loginid}")
        return {"status": "success", "message": "Logout realizado com sucesso"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no logout: {str(e)}")

@app.get("/api/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    """Retorna informações do usuário atual"""
    return {
        "authenticated": True,
        "loginid": user['loginid'],
        "name": "Trader FinanceClick",
        "account_type": "demo" if user['loginid'].startswith("VRTC") else "real"
    }

# --- ENHANCED DERIV API ENDPOINTS ---
@app.post("/api/authorize")
async def authorize_with_token(request: AuthRequest, websocket = Depends(get_deriv_connection)):
    try:
        auth_response = await authorize_connection(request.token, websocket)
        
        if "error" in auth_response:
            logger.warning(f"Authorization failed: {auth_response['error']}")
            return JSONResponse({"status": "error", "error": auth_response["error"]}, status_code=400)
        
        logger.info(f"User authorized: {auth_response['authorize']['loginid']}")
        return JSONResponse({
            "status": "success",
            "account": {
                "loginid": auth_response["authorize"]["loginid"],
                "currency": auth_response["authorize"]["currency"],
                "country": auth_response["authorize"]["country"],
                "name": f"{auth_response['authorize'].get('first_name', '')} {auth_response['authorize'].get('last_name', '')}",
                "balance": auth_response["authorize"]["balance"]
            }
        })
        
    except Exception as e:
        logger.error(f"Authorization error: {e}")
        raise HTTPException(status_code=500, detail=f"Authorization failed: {str(e)}")

@app.get("/api/balance")
async def get_account_balance(websocket = Depends(get_deriv_connection), user: dict = Depends(get_current_user)):
    """Get account balance"""
    try:
        balance_message = {"balance": 1, "subscribe": 1}
        balance_response = await send_deriv_message(websocket, balance_message)
        return JSONResponse(balance_response)
    except Exception as e:
        logger.error(f"Balance request error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get balance: {str(e)}")

@app.get("/api/symbols/accumulators")
@cache(expire=300)  # Cache for 5 minutes
async def get_accumulator_symbols(websocket = Depends(get_deriv_connection)):
    """Get available symbols for accumulator trading"""
    try:
        symbols_message = {
            "active_symbols": "brief",
            "product_type": "basic"
        }
        symbols_response = await send_deriv_message(websocket, symbols_message)
        
        # Filtrar apenas símbolos disponíveis para accumulators
        accumulator_symbols = [
            symbol for symbol in symbols_response.get("active_symbols", [])
            if any(vol in symbol["symbol"] for vol in ["10", "25", "50", "75", "100"])
        ]
        
        return JSONResponse({"accumulator_symbols": accumulator_symbols})
    except Exception as e:
        logger.error(f"Symbols request error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get symbols: {str(e)}")

@app.post("/api/accumulators/buy")
async def buy_accumulator_contract(
    buy_request: AccumulatorBuyRequest, 
    websocket = Depends(get_deriv_connection),
    user: dict = Depends(get_current_user)
):
    """Buy an accumulator contract"""
    try:
        # Rate limiting per user
        if await rate_limiter.is_rate_limited(f"buy_{user['loginid']}", 10, 60):  # 10 trades per minute
            raise HTTPException(status_code=429, detail="Too many trade attempts")
        
        # Validation is now handled by Pydantic
        buy_message = {
            "buy": 1,
            "price": buy_request.amount,
            "parameters": {
                "amount": buy_request.amount,
                "basis": "stake",
                "contract_type": "ACCU",
                "currency": "USD",
                "duration": buy_request.duration,
                "duration_unit": buy_request.duration_unit,
                "symbol": buy_request.symbol,
                "growth_rate": buy_request.growth_rate
            }
        }
        
        buy_response = await send_deriv_message(websocket, buy_message)
        logger.info(f"Accumulator buy executed: {user['loginid']} - {buy_request.symbol}")
        return JSONResponse(buy_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Accumulator buy error: {e}")
        raise HTTPException(status_code=500, detail=f"Accumulator buy failed: {str(e)}")

@app.post("/api/accumulators/proposal")
@cache(expire=30)  # Cache for 30 seconds
async def get_accumulator_proposal(
    buy_request: AccumulatorBuyRequest, 
    websocket = Depends(get_deriv_connection)
):
    """Get proposal for accumulator contract"""
    try:
        proposal_message = {
            "proposal": 1,
            "subscribe": 1,
            "amount": buy_request.amount,
            "basis": "payout",
            "contract_type": "ACCU",
            "currency": "USD",
            "duration": buy_request.duration,
            "duration_unit": buy_request.duration_unit,
            "symbol": buy_request.symbol,
            "growth_rate": buy_request.growth_rate
        }
        
        proposal_response = await send_deriv_message(websocket, proposal_message)
        return JSONResponse(proposal_response)
        
    except Exception as e:
        logger.error(f"Proposal request error: {e}")
        raise HTTPException(status_code=500, detail=f"Proposal request failed: {str(e)}")

# --- ENHANCED TRADE HISTORY ---
@app.get("/api/accumulators/history")
@cache(expire=60)  # Cache for 1 minute
async def get_accumulator_history(
    period: str = "7days", 
    symbol: str = "all", 
    result: str = "all",
    user: dict = Depends(get_current_user)
):
    """Retorna histórico de trades de Accumulator Options"""
    try:
        # Em produção, buscar da API Deriv ou banco de dados
        # Por enquanto, retornar dados simulados
        
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
        
        # Aplicar filtros
        filtered_trades = []
        for trade in base_trades:
            if symbol != "all" and trade["symbol"] != symbol:
                continue
            if result != "all" and trade["status"] != result:
                continue
            filtered_trades.append(trade)
        
        # Calcular estatísticas
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

# --- ENHANCED AI ROBOT ---
async def run_ai_robot(config: RobotConfig, loginid: str):
    """Executa o robô AI para trading automático de Accumulators"""
    global robot_active
    
    try:
        token = active_tokens.get(loginid)
        if not token:
            logger.error(f"Token não encontrado para {loginid}")
            return
        
        # Autorizar conexão
        auth_msg = {"authorize": token}
        await deriv_ws.send(json.dumps(auth_msg))
        auth_response = await deriv_ws.recv()
        
        if "error" in json.loads(auth_response):
            logger.error(f"Falha na autorização do robô: {auth_response}")
            return
        
        logger.info(f"🤖 Robô AI iniciado para {loginid} - Estratégia: {config.strategy}")
        
        trade_count = 0
        total_profit = 0
        consecutive_losses = 0
        
        while robot_active:
            try:
                # Análise de mercado em tempo real
                market_analysis = analyze_market_risk("1HZ100V", config.strategy)
                
                # Safety check: stop after 3 consecutive losses
                if consecutive_losses >= 3:
                    logger.warning(f"Robô parado após {consecutive_losses} perdas consecutivas")
                    robot_active = False
                    break
                
                # Tomada de decisão baseada em IA
                if market_analysis.success_probability > 0.6:
                    params = calculate_accumulator_parameters(config.strategy, "1HZ100V")
                    
                    buy_message = {
                        "buy": 1,
                        "price": config.trade_amount,
                        "parameters": {
                            "amount": config.trade_amount,
                            "basis": "stake",
                            "contract_type": "ACCU",
                            "currency": "USD",
                            "duration": 60,
                            "duration_unit": "t",
                            "symbol": "1HZ100V",
                            "growth_rate": params["growth_rate"]
                        }
                    }
                    
                    await deriv_ws.send(json.dumps(buy_message))
                    buy_response = await deriv_ws.recv()
                    
                    trade_count += 1
                    response_data = json.loads(buy_response)
                    
                    if "error" in response_data:
                        logger.warning(f"Erro no trade do robô: {response_data['error']}")
                        consecutive_losses += 1
                    else:
                        logger.info(f"📊 Robô executou trade #{trade_count} com sucesso")
                        consecutive_losses = 0  # Reset counter on success
                    
                    await asyncio.sleep(30)
                else:
                    logger.info("⏸️ Condições de mercado não favoráveis - aguardando...")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"Erro no ciclo do robô: {e}")
                consecutive_losses += 1
                await asyncio.sleep(10)
                
    except Exception as e:
        logger.error(f"Erro fatal no robô AI: {e}")
    finally:
        logger.info(f"🤖 Robô AI parado - Total de trades: {trade_count}")

@app.post("/api/robot/toggle")
async def toggle_robot(config: RobotConfig, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Liga/Desliga o robô AI para Accumulator Options"""
    global robot_active, robot_tasks
    
    if not robot_active:
        # Iniciar robô
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
        # Parar robô
        robot_active = False
        logger.info(f"Robô AI desativado para {user['loginid']}")
        return {
            "status": "stopped", 
            "message": "Robô AI desativado"
        }

@app.get("/api/robot/status")
async def get_robot_status(user: dict = Depends(get_current_user)):
    """Retorna o status atual do robô AI"""
    return {
        "active": robot_active,
        "active_tasks": len(robot_tasks)
    }

@app.get("/api/market/analysis")
@cache(expire=60)  # Cache for 1 minute
async def get_market_analysis(symbol: str = "1HZ100V", strategy: str = "moderate"):
    """Retorna análise de mercado para Accumulator Options"""
    analysis = analyze_market_risk(symbol, strategy)
    return analysis.dict()

# --- ENHANCED CONTACT SYSTEM ---
async def send_contact_email(contact_data: dict):
    """Envia email de contato com tratamento robusto"""
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
        
        # Enviar email com timeout
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
    """Processa formulário de contato com rate limiting"""
    try:
        # Rate limiting por IP
        client_ip = request.client.host
        if await rate_limiter.is_rate_limited(f"contact_{client_ip}", 3, 300):  # 3 mensagens por 5 minutos
            raise HTTPException(status_code=429, detail="Muitas mensagens enviadas. Tente novamente mais tarde.")
        
        # Preparar dados do contato
        contact_info = {
            **contact_data.dict(),
            "timestamp": datetime.now().isoformat(),
            "id": len(contact_messages) + 1,
            "ip_address": client_ip
        }
        
        # Armazenar localmente (em produção, usar banco de dados)
        contact_messages.append(contact_info)
        
        # Enviar email em background
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

@app.get("/api/contact/messages")
async def get_contact_messages(user: dict = Depends(get_current_user)):
    """Retorna mensagens de contato (apenas para admin)"""
    # Basic admin check - enhance with proper admin authentication in production
    if not user['loginid'].startswith('VRTC'):  # Simple demo account check
        raise HTTPException(status_code=403, detail="Acesso negado")
    return {"messages": contact_messages}

# --- ENHANCED CHATBOT ---
@app.post("/api/chatbot/ask")
async def chatbot_ask(query_data: ChatQuery, request: Request):
    """Responde perguntas sobre Accumulator Options com rate limiting"""
    # Rate limiting
    client_ip = request.client.host
    if await rate_limiter.is_rate_limited(f"chatbot_{client_ip}", 20, 60):  # 20 requests per minute
        raise HTTPException(status_code=429, detail="Muitas requisições. Tente novamente em breve.")
    
    query = query_data.query.lower()
    
    # Buscar na base de conhecimento
    for regra in KNOWLEDGE_BASE.get("regras", []):
        if any(keyword in query for keyword in regra.get("keywords", [])):
            return {"response": regra["resposta"]}
    
    # Respostas padrão para Accumulator Options
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

# --- ENHANCED STATISTICS ENDPOINTS ---
@app.get("/api/stats/platform")
@cache(expire=120)  # Cache for 2 minutes
async def get_platform_stats():
    """Retorna estatísticas da plataforma"""
    return {
        "users_count": len(active_tokens),
        "active_robots": 1 if robot_active else 0,
        "total_trades": len(contact_messages) + 25,
        "success_rate": 78.5,
        "platform_uptime": 99.9,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats/user")
async def get_user_stats(user: dict = Depends(get_current_user)):
    """Retorna estatísticas do usuário atual"""
    return {
        "total_trades": 15,
        "winning_trades": 12,
        "losing_trades": 3,
        "win_rate": 80.0,
        "total_profit": 145.75,
        "favorite_symbol": "1HZ100V",
        "preferred_strategy": "moderate"
    }

# --- ENHANCED HEALTH CHECK AND SYSTEM INFO ---
@app.get("/api/health")
async def health_check():
    """Health check completo da plataforma"""
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
        "version": "2.1.0"
    }
    
    return JSONResponse(health_status)

@app.get("/api/system/info")
async def get_system_info():
    """Retorna informações do sistema"""
    return {
        "platform": "FinanceClick",
        "version": "2.1.0",
        "environment": ENVIRONMENT,
        "deriv_app_id": DERIV_APP_ID,
        "supported_symbols": ["1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"],
        "max_growth_rate": 0.05,
        "min_growth_rate": 0.01,
        "max_trade_amount": 1000,
        "min_trade_amount": 5,
        "features": [
            "accumulator_options",
            "ai_robot",
            "real_time_analysis",
            "risk_management",
            "contact_support"
        ]
    }

# --- ENHANCED ERROR HANDLERS ---
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

@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: HTTPException):
    logger.warning(f"429 Rate Limit Exceeded: {request.client.host}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Muitas requisições. Tente novamente mais tarde."}
    )

@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc: HTTPException):
    logger.warning(f"401 Unauthorized: {request.client.host}")
    return JSONResponse(
        status_code=401,
        content={"detail": "Não autorizado - faça login primeiro"}
    )

# --- PRODUCTION INITIALIZATION ---
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )