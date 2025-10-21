// script.js - FINANCECLICK - VERSÃO FINAL CORRIGIDA
// ✅ CORREÇÕES: Autenticação OAuth robusta + Indicadores visuais

// ==================== PWA SUPPORT ====================
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/service-worker.js')
            .then(function(registration) {
                console.log('✅ Service Worker registrado:', registration.scope);
            })
            .catch(function(error) {
                console.log('❌ Service Worker falhou:', error);
            });
    });
}

// script.js - FINANCECLICK - VERSÃO FINAL CORRIGIDA
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎯 FinanceClick - Inicializando plataforma...');
    
    // ==================== CONFIGURAÇÃO ====================
    const CONFIG = {
        appName: 'FinanceClick',
        version: '2.6.0',
        apiBaseUrl: window.location.origin,
        updateInterval: 30000,
        maxRetries: 3,
        timeout: 10000
    };

    // Estado da aplicação
    let currentUser = null;
    let isRobotActive = false;
    let currentBalance = 0;
    let authCheckInProgress = false;

    // Elementos DOM
    let elements = {};

    // ==================== SISTEMA DE AUTENTICAÇÃO ROBUSTO ====================

    /**
     * ✅ CORREÇÃO CRÍTICA: Processamento OAuth melhorado
     */
    function processOAuthCallback() {
        console.group('🔄 Processamento OAuth Callback - Versão Robusta');
        const urlParams = new URLSearchParams(window.location.search);
        
        const hasOAuthParams = urlParams.has('acct1') || urlParams.has('token1');
        const hasAuthError = urlParams.has('auth_error');
        
        console.log('🔍 Parâmetros detectados:', { 
            hasOAuthParams, 
            hasAuthError,
            authError: urlParams.get('auth_error')
        });
        
        // Tratar erros de autenticação
        if (hasAuthError) {
            const errorCode = urlParams.get('auth_error');
            console.error('❌ Erro de autenticação:', errorCode);
            
            // Limpar URL
            window.history.replaceState({}, document.title, window.location.pathname);
            
            const errorMessages = {
                '1': 'Erro de autenticação na Deriv',
                '2': 'Nenhuma conta recebida',
                '3': 'Erro interno no servidor'
            };
            
            showNotification(errorMessages[errorCode] || 'Erro de autenticação', 'error');
            console.groupEnd();
            return false;
        }
        
        // Processar tokens OAuth
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
                    tokenLength: token ? token.length : 0
                });
                
                if (loginid && token && token.length > 10) {
                    // ✅ CORREÇÃO: Salvar dados com verificação
                    const saved = saveAuthData(loginid, token, currency);
                    if (saved) {
                        tokensProcessed = true;
                        console.log('💾 Tokens salvos:', loginid);
                        
                        // ✅ NOVO: Mostrar indicador visual imediato
                        showAuthSuccessIndicator();
                    }
                    break;
                }
                i++;
            }
            
            if (tokensProcessed) {
                // ✅ CORREÇÃO: Limpar URL antes de qualquer redirecionamento
                window.history.replaceState({}, document.title, window.location.pathname);
                console.log('🧹 URL limpa');
                
                // ✅ CORREÇÃO: Esperar e verificar antes de redirecionar
                setTimeout(() => {
                    if (hasAuthData()) {
                        console.log('🔐 Dados confirmados no localStorage - Redirecionando...');
                        showNotification('Autenticação bem-sucedida! Redirecionando...', 'success');
                        
                        // Pequeno delay para usuário ver a mensagem
                        setTimeout(() => {
                            window.location.href = '/dashboard.html';
                        }, 2000);
                    } else {
                        console.error('❌ Dados não persistidos no localStorage');
                        showNotification('Erro: Falha ao salvar sessão', 'error');
                    }
                }, 1000);
                
                console.groupEnd();
                return true;
            }
        }
        
        console.log('ℹ️ Nenhum token OAuth para processar');
        console.groupEnd();
        return false;
    }

    /**
     * ✅ NOVO: Indicador visual de autenticação bem-sucedida
     */
    function showAuthSuccessIndicator() {
        // Atualizar UI imediatamente
        const tempUser = {
            loginid: 'Carregando...',
            name: 'Usuário FinanceClick',
            account_type: 'real'
        };
        updateUIForAuthenticated(tempUser);
        
        // Mostrar indicador visual
        showNotification('🔐 Sessão restaurada com sucesso!', 'success');
    }

    /**
     * Salvar dados de autenticação com verificação
     */
    function saveAuthData(loginid, token, currency = 'USD') {
        try {
            localStorage.setItem('deriv_token', token);
            localStorage.setItem('deriv_loginid', loginid);
            localStorage.setItem('deriv_currency', currency);
            localStorage.setItem('deriv_last_login', new Date().toISOString());
            
            console.log('🔐 Dados salvos:', loginid);
            return true;
        } catch (error) {
            console.error('❌ Erro ao salvar dados:', error);
            showNotification('Erro ao salvar sessão', 'error');
            return false;
        }
    }

    /**
     * Verificar dados de autenticação
     */
    function hasAuthData() {
        const token = localStorage.getItem('deriv_token');
        const loginid = localStorage.getItem('deriv_loginid');
        const isValid = !!(token && loginid && token.length > 10);
        
        console.log('📋 Verificação auth:', { 
            hasToken: !!token, 
            hasLoginId: !!loginid,
            tokenLength: token?.length,
            isValid 
        });
        
        return isValid;
    }

    /**
     * Limpar dados de autenticação
     */
    function clearAuthData() {
        try {
            localStorage.removeItem('deriv_token');
            localStorage.removeItem('deriv_loginid');
            localStorage.removeItem('deriv_currency');
            localStorage.removeItem('deriv_last_login');
            
            currentUser = null;
            console.log('🧹 Dados removidos');
            return true;
        } catch (error) {
            console.error('❌ Erro ao limpar dados:', error);
            return false;
        }
    }

    /**
     * Obter headers de autenticação
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

    // ==================== INICIALIZAÇÃO ROBUSTA ====================

    /**
     * Inicialização principal
     */
    async function initializeApplication() {
        console.group('🚀 Inicialização da Aplicação - Versão Robusta');
        
        try {
            // 1. Configurações iniciais
            setupEssentialStyles();
            initializeDOMElements();
            setupMobileNavigation();
            setupActiveNavigation();
            
            // 2. ✅ CORREÇÃO: Processar OAuth callback PRIMEIRO
            const hadOAuthCallback = processOAuthCallback();
            
            // 3. Se houve callback, não continuar (já vai redirecionar)
            if (hadOAuthCallback) {
                console.log('🔄 OAuth processado - Aguardando redirecionamento...');
                console.groupEnd();
                return;
            }
            
            // 4. Verificar autenticação existente
            if (hasAuthData()) {
                console.log('🔐 Verificando autenticação existente...');
                await checkAuthentication();
            } else {
                console.log('🔐 Nenhuma autenticação encontrada');
                updateUIForUnauthenticated();
            }
            
            // 5. Configurar interface
            setupUserInterface();
            setupEventListeners();
            await loadInitialData();
            
            // 6. Iniciar serviços
            startBackgroundServices();
            
            console.log('✅ Aplicação inicializada com sucesso');
            
        } catch (error) {
            console.error('💥 Erro na inicialização:', error);
            showNotification('Erro ao inicializar aplicação', 'error');
        }
        
        console.groupEnd();
    }

    /**
     * Inicializar elementos DOM
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
     * ✅ CORREÇÃO: Verificação de autenticação robusta
     */
    async function checkAuthentication() {
        if (authCheckInProgress) {
            console.log('🔐 Verificação de auth já em andamento...');
            return;
        }
        
        authCheckInProgress = true;
        console.group('🔐 Verificação de Autenticação Robusta');
        
        if (!hasAuthData()) {
            console.log('❌ Dados de auth incompletos');
            updateUIForUnauthenticated();
            authCheckInProgress = false;
            console.groupEnd();
            return;
        }

        try {
            console.log('🔄 Consultando /api/me...');
            const response = await fetchWithTimeout('/api/me', {
                headers: getAuthHeaders()
            });

            console.log('📡 Status:', response.status);
            
            if (response.ok) {
                const userData = await response.json();
                console.log('✅ Autenticação válida:', userData.loginid);
                
                currentUser = userData;
                updateUIForAuthenticated(userData);
                
                // ✅ NOVO: Carregar dados adicionais
                await loadUserData();
                
            } else if (response.status === 401) {
                console.log('🔄 Token inválido, tentando restaurar...');
                const restored = await restoreBackendSession();
                
                if (restored) {
                    await checkAuthentication(); // Retry
                } else {
                    console.log('❌ Falha ao restaurar sessão');
                    clearAuthData();
                    updateUIForUnauthenticated();
                    showNotification('Sessão expirada. Faça login novamente.', 'warning');
                }
            } else {
                console.log('❌ Erro na autenticação:', response.status);
                updateUIForUnauthenticated();
            }
            
        } catch (error) {
            console.error('💥 Erro ao verificar auth:', error);
            updateUIForUnauthenticated();
            
            // ✅ NOVO: Tentar restaurar sessão em caso de erro de rede
            if (error.name === 'TypeError') {
                showNotification('Problema de conexão. Tentando reconectar...', 'warning');
                setTimeout(() => checkAuthentication(), 5000);
            }
        }
        
        authCheckInProgress = false;
        console.groupEnd();
    }

    /**
     * ✅ NOVO: Carregar dados do usuário após autenticação
     */
    async function loadUserData() {
        if (!currentUser) return;
        
        try {
            console.log('📥 Carregando dados do usuário...');
            
            await Promise.all([
                updateAccountBalance(),
                updateRobotStatus(),
                loadMarketAnalysis()
            ]);
            
            console.log('✅ Dados do usuário carregados');
            
        } catch (error) {
            console.error('❌ Erro ao carregar dados:', error);
        }
    }

    /**
     * Restaurar sessão no backend
     */
    async function restoreBackendSession() {
        try {
            console.log('🔄 Restaurando sessão no backend...');
            
            const response = await fetchWithTimeout('/api/auth/refresh', {
                method: 'POST',
                headers: getAuthHeaders()
            });

            if (response.ok) {
                console.log('✅ Sessão restaurada');
                return true;
            } else {
                console.log('❌ Falha ao restaurar:', response.status);
                return false;
            }
        } catch (error) {
            console.error('💥 Erro ao restaurar:', error);
            return false;
        }
    }

    // ==================== SISTEMA DE INTERFACE ====================

    /**
     * Atualizar UI para autenticado
     */
    function updateUIForAuthenticated(userData) {
        console.group('🎨 Atualizando UI para Autenticado');
        
        try {
            // Botão login/logout
            if (elements.loginLogoutBtn) {
                elements.loginLogoutBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
                elements.loginLogoutBtn.onclick = handleLogout;
                elements.loginLogoutBtn.className = 'btn-login-logout btn-logout';
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

            // ✅ NOVO: Atualizar status de conexão
            updateConnectionStatus(true);

            console.log('✅ UI atualizada para autenticado');

        } catch (error) {
            console.error('❌ Erro ao atualizar UI:', error);
        }
        
        console.groupEnd();
    }

    /**
     * Atualizar UI para não autenticado
     */
    function updateUIForUnauthenticated() {
        console.group('🎨 Atualizando UI para Não Autenticado');
        
        try {
            // Botão login/logout
            if (elements.loginLogoutBtn) {
                elements.loginLogoutBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
                elements.loginLogoutBtn.onclick = handleLogin;
                elements.loginLogoutBtn.className = 'btn-login-logout btn-login';
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

            // ✅ NOVO: Atualizar status de conexão
            updateConnectionStatus(false);

            console.log('✅ UI atualizada para não autenticado');

        } catch (error) {
            console.error('❌ Erro ao atualizar UI:', error);
        }
        
        console.groupEnd();
    }

    /**
     * ✅ NOVO: Atualizar status de conexão
     */
    function updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (!statusElement) return;

        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-circle" style="color: #28a745"></i><span>Conectado à Deriv API</span>';
            statusElement.style.display = 'flex';
        } else {
            statusElement.style.display = 'none';
        }
    }

    // ==================== SISTEMA DE TRADING ====================

    /**
     * Configurar eventos de trading
     */
    function setupTradingEvents() {
        if (elements.buyAccumulatorBtn) {
            elements.buyAccumulatorBtn.addEventListener('click', executeAccumulatorPurchase);
        }
        if (elements.symbolSelect) {
            elements.symbolSelect.addEventListener('change', updateProposalPreview);
        }
        if (elements.growthRate) {
            elements.growthRate.addEventListener('change', updateProposalPreview);
        }
        if (elements.amount) {
            elements.amount.addEventListener('input', updateProposalPreview);
        }
    }

    /**
     * Executar compra de Accumulator
     */
    async function executeAccumulatorPurchase() {
        if (!currentUser) {
            showNotification('🔐 Faça login antes de negociar', 'warning');
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

            if (tradeData.amount < 5 || tradeData.amount > 1000) {
                showNotification('❌ Valor deve estar entre $5 e $1000', 'error');
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
                showNotification('✅ Compra executada com sucesso!', 'success');
                await updateAccountBalance();
            } else {
                const error = await response.json().catch(() => ({ detail: 'Erro desconhecido' }));
                showNotification(`❌ Erro: ${error.detail || 'Falha na compra'}`, 'error');
            }

        } catch (error) {
            console.error('💥 Erro na compra:', error);
            showNotification('💥 Erro de comunicação', 'error');
        }
    }

    // ==================== SISTEMA FINANCEIRO ====================

    /**
     * Atualizar saldo da conta
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
                    
                    // Efeito visual de atualização
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

    // ==================== UTILITÁRIOS ====================

    /**
     * Fetch com timeout
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
     * Configurar navegação mobile
     */
    function setupMobileNavigation() {
        if (!elements.menuToggle || !elements.mainNav) return;

        elements.menuToggle.addEventListener('click', function() {
            elements.mainNav.classList.toggle('open');
            elements.menuToggle.classList.toggle('active');
        });
    }

    /**
     * Configurar estilos essenciais
     */
    function setupEssentialStyles() {
        if (!document.querySelector('#financeclick-styles')) {
            const styles = document.createElement('style');
            styles.id = 'financeclick-styles';
            styles.textContent = `
                .balance-updated { 
                    animation: pulse 1s;
                    color: #28a745 !important;
                }
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                .account-type {
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.8em;
                    font-weight: bold;
                }
                .account-type.demo { background: #ffeb3b; color: #000; }
                .account-type.real { background: #4CAF50; color: white; }
            `;
            document.head.appendChild(styles);
        }
    }

    // ==================== HANDLERS ====================

    /**
     * Handler de login
     */
    function handleLogin() {
        console.log('🔐 Iniciando login...');
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
        
        try {
            const response = await fetchWithTimeout('/auth/logout', {
                method: 'POST',
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                console.log('✅ Logout no backend');
            }
        } catch (error) {
            console.error('⚠️ Erro no logout:', error);
        } finally {
            clearAuthData();
            updateUIForUnauthenticated();
            showNotification('Logout realizado!', 'success');
            
            if (window.location.pathname.includes('dashboard')) {
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            }
        }
    }

    // ==================== INICIALIZAÇÃO ====================

    /**
     * Configurar event listeners
     */
    function setupEventListeners() {
        setupTradingEvents();
        setupRobotAI();
    }

    /**
     * Configurar interface
     */
    function setupUserInterface() {
        updatePageSpecificUI();
    }

    /**
     * Carregar dados iniciais
     */
    async function loadInitialData() {
        if (!currentUser) return;
        await loadUserData();
    }

    /**
     * Iniciar serviços em background
     */
    function startBackgroundServices() {
        // Atualizar dados a cada 30 segundos
        setInterval(() => {
            if (currentUser) {
                updateAccountBalance();
                updateRobotStatus();
            }
        }, 30000);
    }

    // ==================== SISTEMA DE NOTIFICAÇÕES ====================

    /**
     * Mostrar notificação
     */
    function showNotification(message, type = 'info') {
        // Remover notificações existentes
        document.querySelectorAll('.notification').forEach(n => n.remove());

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${icons[type] || ''}</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;

        // Estilos
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '0',
            borderRadius: '8px',
            color: 'white',
            zIndex: '10000',
            minWidth: '300px',
            maxWidth: '400px',
            animation: 'slideInRight 0.3s ease',
            fontFamily: 'Arial, sans-serif',
            fontSize: '14px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            overflow: 'hidden'
        });

        const colors = {
            success: '#4CAF50',
            error: '#f44336',
            warning: '#ff9800',
            info: '#2196F3'
        };
        
        notification.style.backgroundColor = colors[type] || colors.info;
        document.body.appendChild(notification);

        // Fechar notificação
        notification.querySelector('.notification-close').onclick = () => notification.remove();

        // Auto-remover após 5 segundos
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    // ==================== FUNÇÕES GLOBAIS ====================
    
    window.handleLogin = handleLogin;
    window.handleLogout = handleLogout;
    window.showNotification = showNotification;

    // Inicialização
    function init() {
        console.log(`🚀 ${CONFIG.appName} v${CONFIG.version} - Iniciando...`);
        initializeApplication();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Placeholder functions para evitar erros
    async function updateRobotStatus() {}
    async function loadMarketAnalysis() {}
    function setupRobotAI() {}
    function updateProposalPreview() {}
    function setupActiveNavigation() {}
    function updatePageSpecificUI() {}
});