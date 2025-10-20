// ==================== PWA SUPPORT ====================

// Registrar Service Worker
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/service-worker.js')
            .then(function(registration) {
                console.log('✅ Service Worker registrado com sucesso:', registration.scope);
                
                registration.addEventListener('updatefound', () => {
                    const newWorker = registration.installing;
                    console.log('🔄 Nova versão do Service Worker encontrada');
                    
                    newWorker.addEventListener('statechange', () => {
                        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                            console.log('📦 Nova versão pronta para instalação');
                            showPWAUpdateNotification();
                        }
                    });
                });
            })
            .catch(function(error) {
                console.log('❌ Falha no registro do Service Worker:', error);
            });
    });
}

// Detectar se é PWA
function isRunningAsPWA() {
    return window.matchMedia('(display-mode: standalone)').matches || 
           window.navigator.standalone === true;
}

// Instalar PWA
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    showInstallPromotion();
});

function showInstallPromotion() {
    const installBtn = document.getElementById('installPWA');
    if (installBtn) {
        installBtn.style.display = 'block';
        installBtn.addEventListener('click', installPWA);
    }
}

async function installPWA() {
    if (!deferredPrompt) return;
    
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    
    if (outcome === 'accepted') {
        console.log('✅ PWA instalado pelo usuário');
        showNotification('FinanceClick instalado com sucesso!', 'success');
    } else {
        console.log('❌ Usuário recusou a instalação do PWA');
    }
    
    deferredPrompt = null;
}

function showPWAUpdateNotification() {
    if (isRunningAsPWA()) {
        showNotification('Nova versão disponível! Recarregue o app para atualizar.', 'info');
    }
}

// Network status monitoring
function setupNetworkMonitoring() {
    window.addEventListener('online', () => {
        console.log('✅ Conexão restaurada');
        showNotification('Conexão restaurada - Sincronizando dados...', 'success');
        if (currentUser) {
            updateAccountBalance();
            updateRobotStatus();
        }
    });

    window.addEventListener('offline', () => {
        console.log('❌ Conexão perdida');
        showNotification('Você está offline - Modo limitado', 'warning');
    });
}

// script.js - FINANCECLICK - VERSÃO CORRIGIDA COM FLUXO OAUTH FIXED
document.addEventListener('DOMContentLoaded'), function() {
    console.log('🎯 FinanceClick - Inicializando plataforma de trading...');
}
    // ==================== CONFIGURAÇÃO ====================
    ;const CONFIG = {
        appName: 'FinanceClick',
        version: '2.5.0',
        apiBaseUrl: window.location.origin,
        updateInterval: 30000,
        maxRetries: 3,
        timeout: 10000
    };

    // Estado da aplicação
    let currentUser = null;
    let isRobotActive = false;
    let currentBalance = 0;
    let marketData = null;
    let activeContracts = [];

    // Elementos DOM
    let elements = {};

    // ==================== SISTEMA DE AUTENTICAÇÃO CORRIGIDO ====================

    /**
     * ✅ CORREÇÃO CRÍTICA: Processa callback OAuth da Deriv com redirecionamento correto
     */
    function processOAuthCallback() {
        console.group('🔄 Processamento OAuth Callback - Fluxo Corrigido');
        const urlParams = new URLSearchParams(window.location.search);
        const currentUrl = window.location.href;
        
        console.log('📍 URL atual:', currentUrl);
        
        // Verificar se há parâmetros OAuth OU se há erro de autenticação
        const hasOAuthParams = urlParams.has('acct1') || urlParams.has('token1');
        const hasAuthError = urlParams.has('auth_error');
        
        console.log('🎯 Parâmetros detectados:', { 
            hasOAuthParams, 
            hasAuthError,
            authError: urlParams.get('auth_error')
        });
        
        if (hasAuthError) {
            const errorCode = urlParams.get('auth_error');
            console.error('❌ Erro de autenticação detectado:', errorCode);
            
            // Limpar URL e mostrar mensagem de erro
            const cleanUrl = window.location.pathname;
            window.history.replaceState({}, document.title, cleanUrl);
            
            showNotification('Erro na autenticação. Tente novamente.', 'error');
            console.groupEnd();
            return false;
        }
        
        if (hasOAuthParams) {
            console.log('✅ Iniciando processamento de tokens OAuth...');
            
            let tokensProcessed = false;
            let i = 1;
            
            while (urlParams.has(`acct${i}`) && urlParams.has(`token${i}`)) {
                const loginid = urlParams.get(`acct${i}`);
                const token = urlParams.get(`token${i}`);
                const currency = urlParams.get(`cur${i}`) || 'USD';
                
                console.log(`📥 Processando conta ${i}:`, { 
                    loginid, 
                    token: token ? `***${token.slice(-4)}` : 'NULL'
                });
                
                if (loginid && token && token.length > 10) {
                    saveAuthData(loginid, token, currency);
                    tokensProcessed = true;
                    console.log('💾 Tokens salvos com sucesso:', loginid);
                    break;
                }
                i++;
            }
            
            if (tokensProcessed) {
                // ✅ CORREÇÃO: Limpar URL E redirecionar para dashboard
                const cleanUrl = window.location.pathname;
                window.history.replaceState({}, document.title, cleanUrl);
                console.log('🧹 URL limpa para:', cleanUrl);
                
                // ✅ CORREÇÃO: Redirecionar para dashboard após processar tokens
                console.log('🔄 Redirecionando para dashboard...');
                showNotification('Autenticação bem-sucedida! Redirecionando...', 'success');
                
                // Pequeno delay para mostrar a notificação
                setTimeout(() => {
                    window.location.href = '/dashboard.html';
                }, 1500);
                
                console.groupEnd();
                return true;
            } else {
                console.warn('⚠️ Tokens OAuth presentes mas inválidos');
                showNotification('Erro: Tokens de autenticação inválidos', 'error');
            }
        }
        
        console.log('ℹ️ Nenhum token OAuth para processar');
        console.groupEnd();
        return false;
    }

    /**
     * Salva dados de autenticação no localStorage
     */
    function saveAuthData(loginid, token, currency = 'USD') {
        try {
            localStorage.setItem('deriv_token', token);
            localStorage.setItem('deriv_loginid', loginid);
            localStorage.setItem('deriv_currency', currency);
            localStorage.setItem('deriv_last_login', new Date().toISOString());
            
            console.log('🔐 Dados de autenticação salvos:', loginid);
            return true;
        } catch (error) {
            console.error('❌ Erro ao salvar dados de autenticação:', error);
            return false;
        }
    }

    /**
     * Verifica se existem dados de autenticação
     */
    function hasAuthData() {
        const token = localStorage.getItem('deriv_token');
        const loginid = localStorage.getItem('deriv_loginid');
        const isValid = !!(token && loginid && token.length > 10);
        
        console.log('📋 Verificação dados auth:', { 
            hasToken: !!token, 
            hasLoginId: !!loginid,
            isValid 
        });
        
        return isValid;
    }

    /**
     * Limpa dados de autenticação
     */
    function clearAuthData() {
        try {
            localStorage.removeItem('deriv_token');
            localStorage.removeItem('deriv_loginid');
            localStorage.removeItem('deriv_currency');
            localStorage.removeItem('deriv_last_login');
            
            currentUser = null;
            console.log('🧹 Dados de autenticação removidos');
            return true;
        } catch (error) {
            console.error('❌ Erro ao limpar dados de autenticação:', error);
            return false;
        }
    }

    /**
     * Obtém headers para requisições autenticadas
     */
    function getAuthHeaders() {
        const token = localStorage.getItem('deriv_token');
        const loginid = localStorage.getItem('deriv_loginid');
        
        if (token && loginid) {
            return {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
                'X-LoginID': loginid,
                'X-App-Version': CONFIG.version
            };
        }
        
        return {
            'Content-Type': 'application/json',
            'X-App-Version': CONFIG.version
        };
    }

    // ==================== GESTÃO DE ESTADO DA APLICAÇÃO ====================

    /**
     * Inicialização principal da aplicação
     */
    async function initializeApplication() {
        console.group('🚀 Inicialização da Aplicação - Fluxo Corrigido');
        
        try {
            // 1. Configurações iniciais
            setupEssentialStyles();
            initializeDOMElements();
            setupMobileNavigation();
            setupActiveNavigation();
            
            // 2. ✅ CORREÇÃO: Processar callback OAuth PRIMEIRO (se houver)
            const hadOAuthCallback = processOAuthCallback();
            
            // 3. Se houve callback OAuth, não continuar a inicialização normal
            // pois vamos redirecionar para o dashboard
            if (hadOAuthCallback) {
                console.log('🔄 OAuth processado - Aguardando redirecionamento...');
                console.groupEnd();
                return;
            }
            
            // 4. Verificar autenticação normal (sem OAuth callback)
            if (hasAuthData()) {
                console.log('🔐 Verificando autenticação existente...');
                await checkAuthentication();
            } else {
                console.log('🔐 Nenhum dado de autenticação encontrado');
                updateUIForUnauthenticated();
            }
            
            // 5. Configurar interface e eventos
            setupUserInterface();
            setupEventListeners();
            await loadInitialData();
            
            // 6. Iniciar serviços em background
            startBackgroundServices();
            
            console.log('✅ Aplicação inicializada com sucesso');
            
        } catch (error) {
            console.error('💥 Erro crítico na inicialização:', error);
            showNotification('Erro ao inicializar a aplicação', 'error');
        }
        
        console.groupEnd();
    }

    /**
     * Inicializa elementos DOM de forma robusta
     */
    function initializeDOMElements() {
        const elementIds = [
            'menuToggle', 'mainNav', 'loginLogoutBtn', 'userInfo',
            'accountBalance', 'toggleRobotBtn', 'aiStatus', 'buyAccumulatorBtn',
            'symbolSelect', 'growthRate', 'amount', 'strategySelect',
            'tradeAmount', 'chatInput', 'sendChatBtn', 'chatMessages',
            'contractDetails', 'marketAnalysis', 'proposalResult'
        ];
        
        elements = {};
        elementIds.forEach(id => {
            elements[id] = document.getElementById(id);
            if (!elements[id]) {
                console.warn(`⚠️ Elemento não encontrado: ${id}`);
            }
        });
    }

    /**
     * Verifica autenticação com o backend
     */
    async function checkAuthentication() {
        console.group('🔐 Verificação de Autenticação');
        
        if (!hasAuthData()) {
            console.log('❌ Dados de autenticação incompletos');
            updateUIForUnauthenticated();
            console.groupEnd();
            return;
        }

        try {
            console.log('🔄 Consultando endpoint /api/me...');
            const response = await fetchWithTimeout('/api/me', {
                headers: getAuthHeaders()
            });

            console.log('📡 Status da resposta:', response.status);
            
            if (response.ok) {
                const userData = await response.json();
                console.log('✅ Autenticação válida:', userData.loginid);
                
                currentUser = userData;
                updateUIForAuthenticated(userData);
                
            } else if (response.status === 401) {
                console.log('🔄 Token inválido, tentando restaurar sessão...');
                const restored = await restoreBackendSession();
                
                if (restored) {
                    await checkAuthentication();
                } else {
                    console.log('❌ Falha ao restaurar sessão');
                    clearAuthData();
                    updateUIForUnauthenticated();
                }
            } else {
                console.log('❌ Erro na autenticação:', response.status);
                updateUIForUnauthenticated();
            }
            
        } catch (error) {
            console.error('💥 Erro ao verificar autenticação:', error);
            updateUIForUnauthenticated();
        }
        
        console.groupEnd();
    }

    /**
     * Restaura sessão no backend
     */
    async function restoreBackendSession() {
        try {
            console.log('🔄 Restaurando sessão no backend...');
            
            const response = await fetchWithTimeout('/api/auth/refresh', {
                method: 'POST',
                headers: getAuthHeaders()
            });

            if (response.ok) {
                console.log('✅ Sessão restaurada com sucesso');
                return true;
            } else {
                console.log('❌ Falha ao restaurar sessão:', response.status);
                return false;
            }
        } catch (error) {
            console.error('💥 Erro ao restaurar sessão:', error);
            return false;
        }
    }

    // ==================== SISTEMA DE INTERFACE DO USUÁRIO ====================

    /**
     * Atualiza UI para usuário autenticado
     */
    function updateUIForAuthenticated(userData) {
        console.group('🎨 Atualizando UI para Autenticado');
        
        try {
            // Botão login/logout
            if (elements.loginLogoutBtn) {
                elements.loginLogoutBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
                elements.loginLogoutBtn.onclick = handleLogout;
                elements.loginLogoutBtn.classList.remove('btn-login');
                elements.loginLogoutBtn.classList.add('btn-logout');
            }

            // Informações do usuário
            if (elements.userInfo) {
                elements.userInfo.innerHTML = `
                    <div class="user-welcome">
                        <div class="user-name">Bem-vindo, <strong>${userData.name || 'Trader'}</strong>!</div>
                        <div class="user-account">
                            <span class="account-type ${userData.account_type}">${userData.account_type}</span>
                            <span class="account-id">${userData.loginid}</span>
                        </div>
                    </div>
                `;
                elements.userInfo.style.display = 'flex';
            }

            // Mostrar elementos protegidos
            document.querySelectorAll('[data-auth-required]').forEach(el => {
                el.style.display = 'block';
            });

            // Esconder elementos para não autenticados
            document.querySelectorAll('[data-no-auth]').forEach(el => {
                el.style.display = 'none';
            });

            console.log('✅ UI atualizada para modo autenticado');

        } catch (error) {
            console.error('❌ Erro ao atualizar UI autenticada:', error);
        }
        
        console.groupEnd();
    }

    /**
     * Atualiza UI para usuário não autenticado
     */
    function updateUIForUnauthenticated() {
        console.group('🎨 Atualizando UI para Não Autenticado');
        
        try {
            // Botão login/logout
            if (elements.loginLogoutBtn) {
                elements.loginLogoutBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
                elements.loginLogoutBtn.onclick = handleLogin;
                elements.loginLogoutBtn.classList.remove('btn-logout');
                elements.loginLogoutBtn.classList.add('btn-login');
            }

            // Informações do usuário
            if (elements.userInfo) {
                elements.userInfo.innerHTML = '';
                elements.userInfo.style.display = 'none';
            }

            // Esconder elementos protegidos
            document.querySelectorAll('[data-auth-required]').forEach(el => {
                el.style.display = 'none';
            });

            // Mostrar elementos para não autenticados
            document.querySelectorAll('[data-no-auth]').forEach(el => {
                el.style.display = 'block';
            });

            console.log('✅ UI atualizada para modo não autenticado');

        } catch (error) {
            console.error('❌ Erro ao atualizar UI não autenticada:', error);
        }
        
        console.groupEnd();
    }

    // ==================== SISTEMA DE TRADING ====================

    /**
     * Configura eventos de trading
     */
    function setupTradingEvents() {
        console.group('💰 Configurando Sistema de Trading');
        
        try {
            if (elements.buyAccumulatorBtn) {
                elements.buyAccumulatorBtn.addEventListener('click', executeAccumulatorPurchase);
            }

            if (elements.symbolSelect) {
                elements.symbolSelect.addEventListener('change', handleSymbolChange);
            }

            if (elements.growthRate) {
                elements.growthRate.addEventListener('change', updateProposalPreview);
            }

            if (elements.amount) {
                elements.amount.addEventListener('input', updateProposalPreview);
            }

            console.log('✅ Sistema de trading configurado');

        } catch (error) {
            console.error('❌ Erro ao configurar trading:', error);
        }
        
        console.groupEnd();
    }

    /**
     * Executa compra de Accumulator
     */
    async function executeAccumulatorPurchase() {
        console.group('🛒 Executando Compra Accumulator');
        
        if (!currentUser) {
            showNotification('🔐 Por favor, faça login antes de negociar', 'warning');
            console.groupEnd();
            return;
        }

        try {
            const tradeData = {
                symbol: elements.symbolSelect?.value || '1HZ100V',
                growth_rate: parseFloat(elements.growthRate?.value) || 0.02,
                amount: parseFloat(elements.amount?.value) || 5,
                duration: 60,
                duration_unit: 't'
            };

            console.log('📦 Dados da trade:', tradeData);

            if (tradeData.amount < 5 || tradeData.amount > 1000) {
                showNotification('❌ Valor deve estar entre $5 e $1000', 'error');
                console.groupEnd();
                return;
            }

            showNotification('🔄 Executando compra...', 'info');

            const response = await fetchWithTimeout('/api/accumulators/buy', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify(tradeData)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('✅ Compra executada:', result);
                
                showNotification('✅ Compra executada com sucesso!', 'success');
                
                await updateAccountBalance();
                
                if (result.buy) {
                    displayContractDetails(result.buy);
                }
                
            } else {
                const errorData = await response.json().catch(() => ({ detail: 'Erro desconhecido' }));
                console.error('❌ Erro na compra:', errorData);
                showNotification(`❌ Erro: ${errorData.detail || 'Falha na compra'}`, 'error');
            }

        } catch (error) {
            console.error('💥 Erro na execução da compra:', error);
            showNotification('💥 Erro de comunicação com o servidor', 'error');
        }
        
        console.groupEnd();
    }

    // ==================== SISTEMA DO ROBÔ AI ====================

    /**
     * Configura sistema do robô AI
     */
    function setupRobotAI() {
        console.group('🤖 Configurando Sistema AI');
        
        try {
            if (elements.toggleRobotBtn) {
                elements.toggleRobotBtn.addEventListener('click', toggleRobotAI);
            }

            updateRobotStatus();

            console.log('✅ Sistema AI configurado');

        } catch (error) {
            console.error('❌ Erro ao configurar AI:', error);
        }
        
        console.groupEnd();
    }

    /**
     * Alterna estado do robô AI
     */
    async function toggleRobotAI() {
        if (!currentUser) {
            showNotification('🔐 Faça login para usar o robô AI', 'warning');
            return;
        }

        try {
            const config = {
                strategy: elements.strategySelect?.value || 'moderate',
                trade_amount: parseFloat(elements.tradeAmount?.value) || 5,
                growth_rate: 0.02,
                max_daily_loss: 100,
                take_profit_ticks: 10,
                stop_loss_ticks: 3
            };

            showNotification('🔄 Alternando robô AI...', 'info');

            const response = await fetchWithTimeout('/api/robot/toggle', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                isRobotActive = result.status === 'running';
                
                updateRobotDisplay();
                
                showNotification(
                    isRobotActive ? '🤖 Robô AI ativado!' : '🤖 Robô AI desativado',
                    isRobotActive ? 'success' : 'info'
                );

            } else {
                showNotification('❌ Erro ao controlar robô', 'error');
            }

        } catch (error) {
            console.error('💥 Erro ao alternar robô:', error);
            showNotification('💥 Erro de comunicação', 'error');
        }
    }

    /**
     * Atualiza display do robô
     */
    function updateRobotDisplay() {
        if (!elements.toggleRobotBtn || !elements.aiStatus) return;

        if (isRobotActive) {
            elements.toggleRobotBtn.innerHTML = '<i class="fas fa-stop"></i> PARAR ROBÔ';
            elements.toggleRobotBtn.className = 'btn btn-danger btn-robot-active';
            elements.aiStatus.innerHTML = '<i class="fas fa-circle pulse"></i> ROBÔ ATIVO';
            elements.aiStatus.className = 'robot-status active';
        } else {
            elements.toggleRobotBtn.innerHTML = '<i class="fas fa-play"></i> INICIAR ROBÔ';
            elements.toggleRobotBtn.className = 'btn btn-success btn-robot-inactive';
            elements.aiStatus.innerHTML = '<i class="fas fa-circle"></i> ROBÔ INATIVO';
            elements.aiStatus.className = 'robot-status';
        }
    }

    /**
     * Atualiza status do robô
     */
    async function updateRobotStatus() {
        try {
            const response = await fetchWithTimeout('/api/robot/status', {
                headers: getAuthHeaders()
            });

            if (response.ok) {
                const status = await response.json();
                isRobotActive = status.active;
                updateRobotDisplay();
            }
        } catch (error) {
            console.error('❌ Erro ao verificar status do robô:', error);
        }
    }

    // ==================== SISTEMA FINANCEIRO ====================

    /**
     * Atualiza saldo da conta
     */
    async function updateAccountBalance() {
        if (!elements.accountBalance) return;
        
        try {
            const response = await fetchWithTimeout('/api/balance', {
                headers: getAuthHeaders()
            });

            if (response.ok) {
                const data = await response.json();
                if (data.balance) {
                    currentBalance = data.balance.balance;
                    elements.accountBalance.textContent = 
                        `$${currentBalance.toFixed(2)} ${data.balance.currency || 'USD'}`;
                    
                    elements.accountBalance.classList.add('balance-updated');
                    setTimeout(() => {
                        elements.accountBalance.classList.remove('balance-updated');
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('❌ Erro ao atualizar saldo:', error);
            elements.accountBalance.textContent = 'Erro ao carregar';
        }
    }

    // ==================== UTILITÁRIOS AVANÇADOS ====================

    /**
     * Fetch com timeout e tratamento de erro
     */
    async function fetchWithTimeout(resource, options = {}) {
        const { timeout = CONFIG.timeout } = options;
        
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeout);
        
        try {
            const response = await fetch(resource, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(id);
            return response;
        } catch (error) {
            clearTimeout(id);
            throw error;
        }
    }

    /**
     * Configura navegação mobile
     */
    function setupMobileNavigation() {
        if (!elements.menuToggle || !elements.mainNav) return;

        elements.menuToggle.addEventListener('click', function() {
            elements.mainNav.classList.toggle('open');
            elements.menuToggle.classList.toggle('active');
        });

        const navLinks = elements.mainNav.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth < 768) {
                    elements.mainNav.classList.remove('open');
                    elements.menuToggle.classList.remove('active');
                }
            });
        });
    }

    /**
     * Configura navegação ativa
     */
    function setupActiveNavigation() {
        const mainNav = document.getElementById('mainNav');
        if (!mainNav) return;

        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = mainNav.querySelectorAll('a.nav-link');
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            const linkPage = link.getAttribute('href').split('/').pop();
            
            if (linkPage === currentPage) {
                link.classList.add('active');
            }
        });
    }

    /**
     * Configura estilos essenciais
     */
    function setupEssentialStyles() {
        if (!document.querySelector('#financeclick-styles')) {
            const styles = document.createElement('style');
            styles.id = 'financeclick-styles';
            styles.textContent = `
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                
                .pulse { animation: pulse 2s infinite; }
                .balance-updated { animation: pulse 1s; }
                
                .notification-content {
                    padding: 15px 20px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .notification-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 18px;
                    cursor: pointer;
                    margin-left: auto;
                }
                
                .user-welcome {
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                    text-align: right;
                }
                
                .account-type {
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.8em;
                    font-weight: bold;
                }
                
                .account-type.demo { background: #ffeb3b; color: #000; }
                .account-type.real { background: #4CAF50; color: white; }
                
                .robot-status.active { color: #4CAF50; }
                .robot-status { color: #666; }
            `;
            document.head.appendChild(styles);
        }
    }

    // ==================== HANDLERS DE AUTENTICAÇÃO ====================

    /**
     * Handler de login
     */
    function handleLogin() {
        console.log('🔐 Iniciando processo de login...');
        showNotification('Redirecionando para Deriv...', 'info');
        
        if (elements.loginLogoutBtn) {
            elements.loginLogoutBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Redirecionando...';
            elements.loginLogoutBtn.disabled = true;
        }
        
        setTimeout(() => {
            window.location.href = '/auth/login';
        }, 1000);
    }

    /**
     * Handler de logout
     */
    async function handleLogout() {
        console.log('👋 Iniciando logout...');
    }
        try {
            const response = await fetchWithTimeout('/auth/logout', {
                method: 'POST',
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                console.log('✅ Logout bem-sucedido no backend');
            }
        } catch (error) {
            console.error('⚠️ Erro no logout do backend:', error);
        } finally {
            clearAuthData();
        }