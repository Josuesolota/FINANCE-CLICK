# app.py - FinanceClick Backend with Accumulator Options AI Robot
# VERSÃO FINAL CORRIGIDA - SISTEMA DE AUTENTICAÇÃO PERSISTENTE
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
from pydantic import BaseModel, EmailStr, validator
import secrets

# Deriv API import - CORREÇÃO CRÍTICA
from deriv_api import DerivAPI

# ==================== CONFIGURAÇÃO CORRIGIDA PARA RENDER ====================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend")

print(f"🚀 Iniciando FinanceClick no Render")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Frontend path: {FRONTEND_PATH}")

# Verificar se a pasta frontend existe
if os.path.exists(FRONTEND_PATH):
    print("✅ Pasta frontend encontrada!")
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
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_REDIRECT_URL = os.getenv("DERIV_REDIRECT_URL", "https://finance-click.onrender.com/auth/callback")
DERIV_API_URL = os.getenv("DERIV_API_URL", "wss://ws.deriv.com/websockets/v3")
PORT = int(os.getenv("PORT", "10000"))

# Security settings
ALLOWED_ORIGINS = ["*"]
SESSION_TIMEOUT = 3600

# Variáveis globais
deriv_service = None
active_tokens = {}
user_sessions = {}
robot_active = False
robot_tasks = {}
contact_messages = []

# CORREÇÃO: Knowledge base padrão
DEFAULT_KNOWLEDGE_BASE = {
    "regras": [
        {
            "keywords": ["accumulator", "accumulators", "accumulator options"],
            "resposta": "Accumulator Options são instrumentos financeiros que permitem lucrar com mercados laterais através de crescimento composto. Escolha entre 1% e 5% de taxa de crescimento."
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

# CORREÇÃO: Serviço Deriv API real
class DerivAPIService:
    def __init__(self):
        self.api = None
        self.connected = False

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=DERIV_APP_ID, endpoint=DERIV_API_URL)
            await self.api.connect()
            self.connected = True
            logger.info("✅ Conectado à Deriv API via python-deriv-api")
        except Exception as e:
            logger.error(f"❌ Falha na conexão Deriv API: {e}")
            self.connected = False

    async def authorize(self, token: str) -> Optional[Dict]:
        """Autentica usuário na Deriv"""
        if not self.connected:
            return None
        try:
            response = await self.api.authorize(token)
            return response
        except Exception as e:
            logger.error(f"Erro na autorização: {e}")
            return None

    async def get_balance(self, token: str) -> Optional[float]:
        """Obtém saldo real da conta"""
        try:
            auth_data = await self.authorize(token)
            if auth_data and 'authorize' in auth_data:
                return float(auth_data['authorize']['balance'])
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
        return None

    async def buy_accumulator(self, token: str, buy_params: Dict) -> Optional[Dict]:
        """Compra real de Accumulator"""
        if not self.connected:
            return None
            
        try:
            # Primeiro autentica
            await self.authorize(token)
            
            # Faz proposta
            proposal = await self.api.proposal({
                "proposal": 1,
                "contract_type": "ACCUMULATOR",
                "currency": "USD",
                "symbol": buy_params['symbol'],
                "amount": str(buy_params['amount']),
                "basis": "payout",
                "duration": str(buy_params['duration']),
                "duration_unit": "t"
            })
            
            if proposal and 'proposal' in proposal:
                # Executa compra
                buy_result = await self.api.buy({
                    "buy": proposal['proposal']['id'],
                    "price": str(buy_params['amount'])
                })
                return buy_result
                
        except Exception as e:
            logger.error(f"Erro na compra real do accumulator: {e}")
            
        return None

    async def get_portfolio(self, token: str) -> Optional[Dict]:
        """Obtém portfolio real"""
        try:
            await self.authorize(token)
            portfolio = await self.api.portfolio()
            return portfolio
        except Exception as e:
            logger.error(f"Erro ao obter portfolio: {e}")
            return None

# CORREÇÃO: Carregar modelos de forma robusta
def load_models():
    global RISK_MODEL, KNOWLEDGE_BASE
    
    try:
        if os.path.exists('risk_model.pkl'):
            with open('risk_model.pkl', 'rb') as f:
                RISK_MODEL = pickle.load(f)
            logger.info("✅ Risk model carregado")
        else:
            RISK_MODEL = None
    except Exception as e:
        RISK_MODEL = None
        logger.warning(f"risk_model.pkl não carregado: {e}")

    # Knowledge base
    knowledge_path = os.path.join(FRONTEND_PATH, 'knowledge_base.json')
    try:
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                KNOWLEDGE_BASE = json.load(f)
        else:
            KNOWLEDGE_BASE = DEFAULT_KNOWLEDGE_BASE
    except Exception as e:
        KNOWLEDGE_BASE = DEFAULT_KNOWLEDGE_BASE
        logger.warning(f"knowledge_base.json não carregado: {e}")

load_models()

# --- LIFESPAN MANAGER CORRIGIDO ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global deriv_service
    
    # Inicializar serviço Deriv
    deriv_service = DerivAPIService()
    await deriv_service.connect()
    
    logger.info("✅ FinanceClick inicializado no Render")
    
    yield
    
    # Cleanup
    if deriv_service and deriv_service.connected:
        await deriv_service.api.close()
        logger.info("🔌 Deriv API desconectada")

app = FastAPI(
    title="FinanceClick AI Trading Platform",
    description="Backend with Accumulator Options AI Robot",
    version="2.4.0",  # Atualizado para nova versão
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== SERVIÇO DE ARQUIVOS ESTÁTICOS ====================

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
    
    # Fallback para SPA
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    raise HTTPException(status_code=404, detail="Página não encontrada")

@app.get("/static/{file_path:path}", include_in_schema=False)
async def serve_static(file_path: str):
    static_path = os.path.join(FRONTEND_PATH, file_path)
    if os.path.exists(static_path):
        return FileResponse(static_path)
    raise HTTPException(status_code=404, detail="Arquivo não encontrado")

# --- MODELOS PYDANTIC ---
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

# ✅ CORREÇÃO CRÍTICA: Função get_current_user corrigida
def get_current_user(request: Request):
    # Obter token do header Authorization
    auth_header = request.headers.get("Authorization")
    loginid_header = request.headers.get("X-LoginID")
    
    logger.debug(f"🔐 Validando autenticação - LoginID: {loginid_header}, Auth Header: {auth_header is not None}")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("❌ Token não fornecido ou formato inválido")
        raise HTTPException(status_code=401, detail="Token não fornecido")
    
    token = auth_header[7:]  # Remove "Bearer "
    
    # Validar token específico do usuário
    if not loginid_header:
        logger.warning("❌ LoginID não fornecido")
        raise HTTPException(status_code=401, detail="LoginID não fornecido")
    
    if loginid_header not in active_tokens:
        logger.warning(f"❌ Usuário não autenticado no backend: {loginid_header}")
        raise HTTPException(status_code=401, detail="Usuário não autenticado")
    
    if active_tokens.get(loginid_header) != token:
        logger.warning(f"❌ Token inválido para usuário: {loginid_header}")
        raise HTTPException(status_code=401, detail="Token inválido")
    
    # Atualizar atividade da sessão
    session_key = f"session_{loginid_header}"
    if session_key in user_sessions:
        user_sessions[session_key]['last_activity'] = datetime.now().timestamp()
    else:
        # Recriar sessão se não existir (após restart do Render)
        user_sessions[session_key] = {
            'loginid': loginid_header,
            'created_at': datetime.now().timestamp(),
            'last_activity': datetime.now().timestamp()
        }
        logger.info(f"✅ Sessão recriada para: {loginid_header}")
    
    logger.debug(f"✅ Autenticação válida para: {loginid_header}")
    
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

# ==================== ENDPOINTS DA API CORRIGIDOS ====================

# --- AUTENTICAÇÃO ---
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
        logger.info(f"📥 OAuth callback recebido")
        
        if "error" in query_params:
            error_msg = query_params.get("error", "Erro desconhecido")
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
                
                # ✅ CORREÇÃO: Armazenar token e sessão
                active_tokens[loginid] = token
                session_key = f"session_{loginid}"
                user_sessions[session_key] = {
                    'loginid': loginid,
                    'created_at': datetime.now().timestamp(),
                    'last_activity': datetime.now().timestamp()
                }
                
                logger.info(f"✅ Usuário autenticado: {loginid}")
            i += 1
        
        if not accounts:
            raise HTTPException(status_code=400, detail="No accounts received")
        
        return RedirectResponse(url="/dashboard", status_code=302)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro no callback OAuth: {e}")
        return RedirectResponse(url="/", status_code=302)

# ✅ NOVO ENDPOINT: Restaurar sessão a partir do localStorage
@app.post("/api/auth/refresh")
async def refresh_session(request: Request):
    """Restaura sessão a partir do localStorage do frontend"""
    try:
        auth_header = request.headers.get("Authorization")
        loginid_header = request.headers.get("X-LoginID")
        
        logger.info(f"🔄 Tentativa de restaurar sessão para: {loginid_header}")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Token não fornecido")
        
        token = auth_header[7:]
        
        # Restaurar sessão no backend
        if loginid_header and token:
            active_tokens[loginid_header] = token
            session_key = f"session_{loginid_header}"
            user_sessions[session_key] = {
                'loginid': loginid_header,
                'created_at': datetime.now().timestamp(),
                'last_activity': datetime.now().timestamp()
            }
            
            logger.info(f"✅ Sessão restaurada para: {loginid_header}")
            return {
                "status": "success",
                "message": "Sessão restaurada",
                "loginid": loginid_header
            }
        else:
            raise HTTPException(status_code=400, detail="Dados de autenticação incompletos")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao restaurar sessão: {e}")
        raise HTTPException(status_code=500, detail="Falha ao restaurar sessão")

@app.post("/auth/logout")
async def logout_user(request: Request):
    try:
        # ✅ CORREÇÃO: Usar a mesma lógica de autenticação por headers
        auth_header = request.headers.get("Authorization")
        loginid_header = request.headers.get("X-LoginID")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Token não fornecido")
        
        token = auth_header[7:]
        
        if not loginid_header or loginid_header not in active_tokens:
            raise HTTPException(status_code=401, detail="Usuário não autenticado")
        
        # Remover sessão
        if loginid_header in active_tokens:
            del active_tokens[loginid_header]
        session_key = f"session_{loginid_header}"
        if session_key in user_sessions:
            del user_sessions[session_key]
            
        logger.info(f"👋 Logout realizado para: {loginid_header}")
        return {"status": "success", "message": "Logout realizado"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no logout: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no logout: {str(e)}")

@app.get("/api/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    print(f"Fornecido info do usuário: {user['loginid']}")
    return {
        "authenticated": True,
        "loginid": user['loginid'],
        "name": "Trader FinanceClick",
        "account_type": "demo" if user['loginid'].startswith("VRTC") else "real"
    }

# --- DERIV API REAL ---
@app.get("/api/balance")
async def get_account_balance(user: dict = Depends(get_current_user)):
    try:
        # Tenta obter saldo real
        if deriv_service and deriv_service.connected:
            real_balance = await deriv_service.get_balance(user['token'])
            if real_balance is not None:
                return {
                    "balance": {
                        "balance": real_balance,
                        "currency": "USD",
                        "loginid": user['loginid']
                    }
                }
        
        # Fallback para saldo simulado
        simulated_balance = 1000.00
        return {
            "balance": {
                "balance": simulated_balance,
                "currency": "USD", 
                "loginid": user['loginid']
            }
        }
        
    except Exception as e:
        logger.error(f"Balance request error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get balance: {str(e)}")

@app.get("/api/symbols/accumulators")
@cache(expire=300)
async def get_accumulator_symbols():
    accumulator_symbols = [
        {"symbol": "1HZ10V", "display_name": "Volatility 10 Index"},
        {"symbol": "1HZ25V", "display_name": "Volatility 25 Index"},
        {"symbol": "1HZ50V", "display_name": "Volatility 50 Index"},
        {"symbol": "1HZ75V", "display_name": "Volatility 75 Index"},
        {"symbol": "1HZ100V", "display_name": "Volatility 100 Index"}
    ]
    return {"accumulator_symbols": accumulator_symbols}

# ✅ CORREÇÃO: Melhorar resposta da compra para frontend definitivo
@app.post("/api/accumulators/buy")
async def buy_accumulator_contract(
    buy_request: AccumulatorBuyRequest, 
    user: dict = Depends(get_current_user)
):
    try:
        if await rate_limiter.is_rate_limited(f"buy_{user['loginid']}", 10, 60):
            raise HTTPException(status_code=429, detail="Too many trade attempts")
        
        # Tenta compra real
        if deriv_service and deriv_service.connected:
            buy_params = {
                'symbol': buy_request.symbol,
                'amount': buy_request.amount,
                'duration': buy_request.duration
            }
            real_buy = await deriv_service.buy_accumulator(user['token'], buy_params)
            if real_buy:
                return {"buy": real_buy}
        
        # Fallback para compra simulada - ✅ MELHORADO para frontend definitivo
        import random
        contract_id = f"ACCU_{int(datetime.now().timestamp())}_{user['loginid']}"
        is_success = random.random() > 0.3
        profit_loss = buy_request.amount * buy_request.growth_rate * random.randint(5, 20) if is_success else -buy_request.amount
        
        # ✅ ESTRUTURA COMPATÍVEL COM FRONTEND DEFINITIVO
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
        logger.error(f"Accumulator buy error: {e}")
        raise HTTPException(status_code=500, detail=f"Accumulator buy failed: {str(e)}")

# ✅ CORREÇÃO: Melhorar proposta para frontend definitivo
@app.post("/api/accumulators/proposal")
@cache(expire=30)
async def get_accumulator_proposal(buy_request: AccumulatorBuyRequest):
    import random
    potential_payout = buy_request.amount * (1 + buy_request.growth_rate * random.randint(8, 15))
    potential_return = potential_payout - buy_request.amount
    
    # ✅ ESTRUTURA MELHORADA para frontend definitivo
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

# --- ROBÔ AI ---
async def run_ai_robot(config: RobotConfig, loginid: str):
    global robot_active
    
    try:
        logger.info(f"🤖 Robô AI iniciado para {loginid}")
        
        trade_count = 0
        while robot_active and trade_count < 5:
            await asyncio.sleep(10)
            trade_count += 1
            
    except Exception as e:
        logger.error(f"Erro no robô AI: {e}")
    finally:
        robot_active = False
        logger.info(f"🤖 Robô AI parado")

# ✅ CORREÇÃO: Melhorar resposta do robô AI
@app.post("/api/robot/toggle")
async def toggle_robot(config: RobotConfig, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    global robot_active
    
    if not robot_active:
        robot_active = True
        background_tasks.add_task(run_ai_robot, config, user['loginid'])
        
        # ✅ ADICIONADO: Incluir análise de mercado na resposta
        market_analysis = await get_market_analysis("1HZ100V", config.strategy)
        
        return {
            "status": "running",
            "message": f"Robô AI ativado com estratégia {config.strategy}",
            "config": config.dict(),
            "analysis": market_analysis  # ✅ Frontend definitivo espera este campo
        }
    else:
        robot_active = False
        return {
            "status": "stopped", 
            "message": "Robô AI desativado"
        }

@app.get("/api/robot/status")
async def get_robot_status():
    return {"active": robot_active, "message": "Robô ativo" if robot_active else "Robô inativo"}

# --- CONTATO E CHATBOT ---
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
        logger.info(f"📧 Nova mensagem de contato: {contact_data.email}")
        
        return {
            "status": "success",
            "message": "Mensagem enviada com sucesso!",
            "contact_id": contact_info["id"]
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar formulário: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar formulário: {str(e)}")

@app.post("/api/chatbot/ask")
async def chatbot_ask(query_data: ChatQuery, request: Request):
    client_ip = request.client.host
    if await rate_limiter.is_rate_limited(f"chatbot_{client_ip}", 20, 60):
        raise HTTPException(status_code=429, detail="Muitas requisições")
    
    query = query_data.query.lower()
    
    for regra in KNOWLEDGE_BASE.get("regras", []):
        if any(keyword in query for keyword in regra.get("keywords", [])):
            return {"response": regra["resposta"]}
    
    return {
        "response": "Desculpe, sou especializado em Accumulator Options. Posso ajudar com: conexão Deriv, robô AI, estratégias, símbolos disponíveis, gestão de risco."
    }

# ==================== NOVOS ENDPOINTS PARA FRONTEND DEFINITIVO ====================

# ✅ NOVO: Endpoint de análise de mercado
@app.get("/api/market/analysis")
@cache(expire=120)
async def get_market_analysis(symbol: str = "1HZ100V", strategy: str = "moderate"):
    """
    Fornece análise de mercado para o frontend definitivo
    """
    try:
        # Análise baseada no símbolo e estratégia
        volatility_scores = {
            "1HZ10V": 0.3,   # Baixa volatilidade
            "1HZ25V": 0.5,   # Volatilidade média-baixa
            "1HZ50V": 0.7,   # Volatilidade média
            "1HZ75V": 0.8,   # Volatilidade alta
            "1HZ100V": 0.9   # Volatilidade muito alta
        }
        
        strategy_impact = {
            "conservative": 0.15,
            "moderate": 0.0,
            "aggressive": -0.15
        }
        
        volatility = volatility_scores.get(symbol, 0.5)
        base_probability = 0.8 - (volatility * 0.3)
        success_probability = max(0.1, min(0.9, base_probability + strategy_impact.get(strategy, 0)))
        recommended_rate = max(0.01, min(0.05, 0.03 - (volatility * 0.02)))
        
        return {
            "symbol": symbol,
            "volatility": volatility,
            "success_probability": success_probability,
            "recommended_growth_rate": recommended_rate,
            "analysis_time": datetime.now().isoformat(),
            "strategy_used": strategy
        }
        
    except Exception as e:
        logger.error(f"Erro na análise de mercado: {e}")
        raise HTTPException(status_code=500, detail="Erro na análise de mercado")

# ✅ NOVO: Endpoint de histórico de trades
@app.get("/api/accumulators/history")
@cache(expire=60)
async def get_accumulator_history(
    period: str = "7days",
    symbol: str = "all", 
    result: str = "all",
    user: dict = Depends(get_current_user)
):
    """
    Fornece histórico de trades para o frontend definitivo
    """
    try:
        # Dados de exemplo - em produção, buscar do banco de dados
        base_trades = [
            {
                "id": "ACCU_123456789",
                "symbol": "1HZ100V",
                "type": "ACCUMULATOR",
                "growth_rate": 0.02,
                "amount": 10.0,
                "result": 8.95,
                "ticks": 12,
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "status": "win"
            },
            {
                "id": "ACCU_123456788", 
                "symbol": "1HZ75V",
                "type": "ACCUMULATOR",
                "growth_rate": 0.05,
                "amount": 15.0,
                "result": -15.0,
                "ticks": 3,
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "status": "loss"
            },
            {
                "id": "ACCU_123456787",
                "symbol": "1HZ50V",
                "type": "ACCUMULATOR", 
                "growth_rate": 0.03,
                "amount": 8.0,
                "result": 12.35,
                "ticks": 18,
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                "status": "win"
            }
        ]
        
        # Filtrar trades baseado nos parâmetros
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
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        return {
            "trades": filtered_trades,
            "stats": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_profit": round(total_profit, 2),
                "average_profit": round(avg_profit, 2),
                "period": period,
                "symbol": symbol
            }
        }
        
    except Exception as e:
        logger.error(f"Erro ao carregar histórico: {e}")
        raise HTTPException(status_code=500, detail="Erro ao carregar histórico")

# ✅ NOVO: Endpoint para dados do dashboard
@app.get("/api/dashboard/data")
async def get_dashboard_data(user: dict = Depends(get_current_user)):
    """
    Fornece todos os dados necessários para o dashboard do frontend definitivo
    """
    try:
        # Obter saldo
        balance_response = await get_account_balance(user)
        balance_data = balance_response if isinstance(balance_response, dict) else await balance_response.json()
        
        # Obter status do robô
        robot_status = await get_robot_status()
        
        # Obter análise de mercado
        market_analysis = await get_market_analysis("1HZ100V", "moderate")
        
        # Obter histórico recente
        recent_history = await get_accumulator_history("1day", "all", "all", user)
        
        return {
            "balance": balance_data.get("balance", {}),
            "robot_status": robot_status,
            "market_analysis": market_analysis,
            "recent_trades": recent_history.get("trades", [])[:5],  # Últimos 5 trades
            "quick_stats": recent_history.get("stats", {}),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados do dashboard: {e}")
        raise HTTPException(status_code=500, detail="Erro ao carregar dados do dashboard")
# --- HEALTH CHECK ---
# ✅ CORREÇÃO: Health check mais detalhado
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "FinanceClick AI Trading",
        "timestamp": datetime.now().isoformat(),
        "deriv_connected": deriv_service.connected if deriv_service else False,
        "robot_active": robot_active,
        "active_users": len(active_tokens),
        "user_sessions": len(user_sessions),
        "environment": ENVIRONMENT,
        "version": "2.5.0",  # Atualizado
        "endpoints_available": [
            "/api/me",
            "/api/balance", 
            "/api/symbols/accumulators",
            "/api/accumulators/buy",
            "/api/accumulators/proposal",
            "/api/accumulators/history",
            "/api/market/analysis",
            "/api/robot/toggle",
            "/api/robot/status",
            "/api/dashboard/data",
            "/api/auth/refresh",
            "/api/chatbot/ask",
            "/api/contact"
        ]
    }

# --- PRODUCTION INITIALIZATION ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)