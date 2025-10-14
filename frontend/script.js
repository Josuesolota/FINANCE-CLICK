// script.js - FINANCECLICK - VERSÃO COMPLETA E ROBUSTA
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎯 FinanceClick - Inicializando plataforma de trading...');
    
    // ==================== CONFIGURAÇÃO E VARIÁVEIS ====================
    const CONFIG = {
        appName: 'FinanceClick',
        version: '2.0.0',
        apiBaseUrl: '',
        updateInterval: 30000,
        maxRetries: 3
    };

    // Estado da aplicação
    let currentUser = null;
    let isRobotActive = false;
    let currentBalance = 0;
    let marketData = null;
    let activeContracts = [];
    let retryCount = 0;

    // Elementos DOM - Busca robusta
    const elements = {
        menuToggle: document.getElementById('menuToggle'),
        mainNav: document.getElementById('mainNav'),
        loginLogoutBtn: document.getElementById('loginLogoutBtn'),
        userInfoElement: document.getElementById('userInfo'),
        accountBalance: document.getElementById('accountBalance'),
        toggleRobotBtn: document.getElementById('toggleRobotBtn'),
        aiStatus: document.getElementById('aiStatus'),
        buyAccumulatorBtn: document.getElementById('buyAccumulatorBtn'),
        symbolSelect: document.getElementById('symbolSelect'),
        growthRate: document.getElementById('growthRate'),
        amount: document.getElementById('amount'),
        strategySelect: document.getElementById('strategySelect'),
        tradeAmount: document.getElementById('tradeAmount'),
        chatInput: document.getElementById('chatInput'),
        sendChatBtn: document.getElementById('sendChatBtn'),
        chatMessages: document.getElementById('chatMessages'),
        contractDetails: document.getElementById('contractDetails'),
        marketAnalysis: document.getElementById('marketAnalysis'),
        proposalResult: document.getElementById('proposalResult')
    };

    // ==================== SISTEMA DE AUTENTICAÇÃO ====================

    /**
     * Processa callback OAuth da Deriv - CORREÇÃO CRÍTICA
     */
    function processOAuthCallback() {
        console.group('🔄 Processamento OAuth Callback');
        const urlParams = new URLSearchParams(window.location.search);
        const currentUrl = window.location.href;
        
        console.log('📍 URL completa:', currentUrl);
        console.log('📋 Parâmetros URL:', window.location.search);
        
        const hasOAuthParams = urlParams.has('acct1') || urlParams.has('token1');
        console.log('🎯 Parâmetros OAuth detectados:', hasOAuthParams);
        
        if (hasOAuthParams) {
            console.log('✅ Iniciando processamento de tokens OAuth...');
            
            let tokensProcessed = false;
            let i = 1;
            
            // Processar todas as contas retornadas
            while (urlParams.has(`acct${i}`) && urlParams.has(`token${i}`)) {
                const loginid = urlParams.get(`acct${i}`);
                const token = urlParams.get(`token${i}`);
                const currency = urlParams.get(`cur${i}`) || 'USD';
                
                console.log(`📥 Processando conta ${i}:`, { 
                    loginid, 
                    token: token ? `***${token.slice(-4)}` : 'NULL', 
                    currency 
                });
                
                if (loginid && token && token.length > 10) {
                    // Salvar dados de autenticação
                    saveAuthData(loginid, token, currency);
                    tokensProcessed = true;
                    
                    console.log('💾 Tokens salvos com sucesso:', loginid);
                    break; // Usar primeira conta válida
                } else {
                    console.warn('⚠️ Token ou loginid inválido na conta', i);
                }
                i++;
            }
            
            if (tokensProcessed) {
                // Limpar URL parameters - IMPORTANTE!
                const cleanUrl = window.location.pathname;
                window.history.replaceState({}, document.title, cleanUrl);
                console.log('🧹 URL limpa para:', cleanUrl);
                
                console.groupEnd();
                return true;
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
            tokenLength: token?.length,
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
        console.group('🚀 Inicialização da Aplicação');
        
        try {
            // 1. Configurações iniciais
            setupEssentialStyles();
            setupMobileNavigation();
            setupActiveNavigation();
            
            // 2. Processar callback OAuth (se houver)
            const hadOAuthCallback = processOAuthCallback();
            
            // 3. Verificar autenticação
            if (hadOAuthCallback || hasAuthData()) {
                console.log('🔐 Verificando autenticação...');
                await checkAuthentication();
                
                if (hadOAuthCallback) {
                    await restoreBackendSession();
                    // Re-verificar após restaurar sessão
                    await checkAuthentication();
                }
            } else {
                console.log('🔐 Nenhum dado de autenticação encontrado');
                updateUIForUnauthenticated();
            }
            
            // 4. Configurar interface e eventos
            setupUserInterface();
            setupEventListeners();
            await loadInitialData();
            
            // 5. Iniciar serviços em background
            startBackgroundServices();
            
            console.log('✅ Aplicação inicializada com sucesso');
            
        } catch (error) {
            console.error('💥 Erro crítico na inicialização:', error);
            showNotification('Erro ao inicializar a aplicação', 'error');
        }
        
        console.groupEnd();
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
            const response = await fetch('/api/me', {
                headers: getAuthHeaders(),
                timeout: 10000
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
                    await checkAuthentication(); // Retry
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
            
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: getAuthHeaders(),
                timeout: 10000
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
            if (elements.userInfoElement) {
                elements.userInfoElement.innerHTML = `
                    <div class="user-welcome">
                        <div class="user-name">Bem-vindo, <strong>${userData.name || 'Trader'}</strong>!</div>
                        <div class="user-account">
                            <span class="account-type ${userData.account_type}">${userData.account_type}</span>
                            <span class="account-id">${userData.loginid}</span>
                        </div>
                    </div>
                `;
                elements.userInfoElement.style.display = 'flex';
            }

            // Mostrar elementos protegidos
            document.querySelectorAll('[data-auth-required]').forEach(el => {
                el.style.display = 'block';
                el.classList.add('authenticated-visible');
            });

            // Esconder elementos para não autenticados
            document.querySelectorAll('[data-no-auth]').forEach(el => {
                el.style.display = 'none';
            });

            // Atualizar elementos específicos da página
            updatePageSpecificUI();

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
            if (elements.userInfoElement) {
                elements.userInfoElement.innerHTML = '';
                elements.userInfoElement.style.display = 'none';
            }

            // Esconder elementos protegidos
            document.querySelectorAll('[data-auth-required]').forEach(el => {
                el.style.display = 'none';
                el.classList.remove('authenticated-visible');
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

    /**
     * Atualiza UI específica da página atual
     */
    function updatePageSpecificUI() {
        const currentPage = window.location.pathname;
        
        if (currentPage.includes('dashboard')) {
            updateDashboardUI();
        } else if (currentPage.includes('history')) {
            updateHistoryUI();
        } else if (currentPage.includes('guide')) {
            updateGuideUI();
        }
    }

    /**
     * Atualiza UI do dashboard
     */
    function updateDashboardUI() {
        console.log('📊 Atualizando UI do dashboard...');
        
        // Atualizar saldo
        updateAccountBalance();
        
        // Atualizar status do robô
        updateRobotStatus();
        
        // Carregar análise de mercado
        loadMarketAnalysis();
        
        // Carregar símbolos disponíveis
        loadTradingSymbols();
    }

    // ==================== SISTEMA DE TRADING ====================

    /**
     * Configura eventos de trading
     */
    function setupTradingEvents() {
        console.group('💰 Configurando Sistema de Trading');
        
        try {
            // Botão de compra
            if (elements.buyAccumulatorBtn) {
                elements.buyAccumulatorBtn.addEventListener('click', executeAccumulatorPurchase);
                console.log('✅ Botão de compra configurado');
            }

            // Select de símbolos
            if (elements.symbolSelect) {
                elements.symbolSelect.addEventListener('change', handleSymbolChange);
            }

            // Select de taxa de crescimento
            if (elements.growthRate) {
                elements.growthRate.addEventListener('change', updateProposalPreview);
            }

            // Input de valor
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

            // Validar dados
            if (tradeData.amount < 5 || tradeData.amount > 1000) {
                showNotification('❌ Valor deve estar entre $5 e $1000', 'error');
                console.groupEnd();
                return;
            }

            showNotification('🔄 Executando compra...', 'info');

            const response = await fetch('/api/accumulators/buy', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify(tradeData)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('✅ Compra executada:', result);
                
                showNotification('✅ Compra executada com sucesso!', 'success');
                
                // Atualizar dados
                await updateAccountBalance();
                await loadTradingHistory();
                
                // Mostrar detalhes do contrato
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

    /**
     * Carrega símbolos de trading disponíveis
     */
    async function loadTradingSymbols() {
        if (!elements.symbolSelect) return;
        
        try {
            console.log('📈 Carregando símbolos...');
            
            const response = await fetch('/api/symbols/accumulators', {
                headers: getAuthHeaders()
            });

            if (response.ok) {
                const data = await response.json();
                elements.symbolSelect.innerHTML = '';
                
                data.accumulator_symbols.forEach(symbol => {
                    const option = document.createElement('option');
                    option.value = symbol.symbol;
                    option.textContent = `${symbol.display_name} (${symbol.symbol})`;
                    elements.symbolSelect.appendChild(option);
                });
                
                console.log(`✅ ${data.accumulator_symbols.length} símbolos carregados`);
            }
        } catch (error) {
            console.error('❌ Erro ao carregar símbolos:', error);
        }
    }

    /**
     * Atualiza preview da proposta
     */
    async function updateProposalPreview() {
        if (!elements.proposalResult) return;
        
        try {
            const proposalData = {
                symbol: elements.symbolSelect?.value || '1HZ100V',
                growth_rate: parseFloat(elements.growthRate?.value) || 0.02,
                amount: parseFloat(elements.amount?.value) || 5,
                duration: 60,
                duration_unit: 't'
            };

            const response = await fetch('/api/accumulators/proposal', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify(proposalData)
            });

            if (response.ok) {
                const data = await response.json();
                elements.proposalResult.innerHTML = `
                    <div class="proposal-card">
                        <h4>📊 Proposta Calculada</h4>
                        <div class="proposal-details">
                            <p><strong>Payout Potencial:</strong> $${data.proposal?.display_value || '0.00'}</p>
                            <p><strong>Taxa de Crescimento:</strong> ${(proposalData.growth_rate * 100).toFixed(1)}%</p>
                            <p><strong>Retorno Estimado:</strong> $${(parseFloat(data.proposal?.payout || 0) - proposalData.amount).toFixed(2)}</p>
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('❌ Erro ao obter proposta:', error);
        }
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
                console.log('✅ Botão do robô configurado');
            }

            if (elements.strategySelect) {
                elements.strategySelect.addEventListener('change', updateRobotStrategy);
            }

            if (elements.tradeAmount) {
                elements.tradeAmount.addEventListener('input', updateTradeAmountDisplay);
            }

            // Verificar status atual
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

            const response = await fetch('/api/robot/toggle', {
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
            const response = await fetch('/api/robot/status', {
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
            const response = await fetch('/api/balance', {
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

    /**
     * Carrega histórico de trades
     */
    async function loadTradingHistory() {
        if (!window.location.pathname.includes('history')) return;
        
        try {
            const response = await fetch('/api/accumulators/history', {
                headers: getAuthHeaders()
            });

            if (response.ok) {
                const data = await response.json();
                updateHistoryDisplay(data);
            }
        } catch (error) {
            console.error('❌ Erro ao carregar histórico:', error);
        }
    }

    // ==================== SISTEMA DE ANÁLISE DE MERCADO ====================

    /**
     * Carrega análise de mercado
     */
    async function loadMarketAnalysis() {
        if (!elements.marketAnalysis) return;
        
        try {
            const response = await fetch('/api/market/analysis?symbol=1HZ100V&strategy=moderate', {
                headers: getAuthHeaders()
            });

            if (response.ok) {
                const analysis = await response.json();
                displayMarketAnalysis(analysis);
            }
        } catch (error) {
            console.error('❌ Erro ao carregar análise:', error);
        }
    }

    /**
     * Exibe análise de mercado
     */
    function displayMarketAnalysis(analysis) {
        if (!elements.marketAnalysis) return;

        const volatilityClass = analysis.volatility > 0.7 ? 'high' : 
                              analysis.volatility > 0.4 ? 'medium' : 'low';
        
        const probabilityClass = analysis.success_probability > 0.7 ? 'high' : 
                                analysis.success_probability > 0.4 ? 'medium' : 'low';

        elements.marketAnalysis.innerHTML = `
            <div class="analysis-card">
                <h4>📈 Análise do Mercado</h4>
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <label>Volatilidade:</label>
                        <span class="metric ${volatilityClass}">
                            ${(analysis.volatility * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="analysis-item">
                        <label>Prob. Sucesso:</label>
                        <span class="metric ${probabilityClass}">
                            ${(analysis.success_probability * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="analysis-item">
                        <label>Taxa Recomendada:</label>
                        <span class="metric recommended">
                            ${(analysis.recommended_growth_rate * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>
        `;
    }

    // ==================== SISTEMA DE CHATBOT ====================

    /**
     * Configura sistema de chatbot
     */
    function setupChatbot() {
        if (!elements.chatInput || !elements.sendChatBtn || !elements.chatMessages) {
            return;
        }

        console.log('💬 Configurando chatbot...');

        elements.sendChatBtn.addEventListener('click', sendChatMessage);
        elements.chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });

        // Mensagem de boas-vindas
        addChatMessage('Olá! Sou o assistente da FinanceClick. Como posso ajudar você com Accumulator Options?', false);
    }

    /**
     * Envia mensagem no chatbot
     */
    async function sendChatMessage() {
        const message = elements.chatInput.value.trim();
        if (!message) return;

        addChatMessage(message, true);
        elements.chatInput.value = '';

        try {
            const response = await fetch('/api/chatbot/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            });

            if (response.ok) {
                const data = await response.json();
                addChatMessage(data.response, false);
            } else {
                addChatMessage('Desculpe, houve um erro ao processar sua mensagem.', false);
            }
        } catch (error) {
            console.error('❌ Erro no chatbot:', error);
            addChatMessage('Desculpe, estou com problemas de conexão no momento.', false);
        }
    }

    /**
     * Adiciona mensagem ao chat
     */
    function addChatMessage(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${isUser ? 'user-message' : 'bot-message'}`;
        
        if (isUser) {
            messageDiv.innerHTML = `
                <div class="message-content user">
                    <div class="message-text">${message}</div>
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content bot">
                    <div class="bot-avatar">🤖</div>
                    <div class="message-text">${message}</div>
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
        }

        elements.chatMessages.appendChild(messageDiv);
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    // ==================== SISTEMA DE NOTIFICAÇÕES ====================

    /**
     * Mostra notificação para o usuário
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
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    &times;
                </button>
            </div>
        `;

        // Aplicar estilos
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

        // Auto-remover após 5 segundos
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    // ==================== UTILITÁRIOS DE INTERFACE ====================

    /**
     * Configura navegação mobile
     */
    function setupMobileNavigation() {
        if (!elements.menuToggle || !elements.mainNav) return;

        elements.menuToggle.addEventListener('click', function() {
            elements.mainNav.classList.toggle('open');
            elements.menuToggle.classList.toggle('active');
        });

        // Fechar menu ao clicar em links (mobile)
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
                
                .metric.high { color: #f44336; }
                .metric.medium { color: #ff9800; }
                .metric.low { color: #4CAF50; }
                .metric.recommended { color: #2196F3; }
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
        
        // Feedback visual
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
            const response = await fetch('/auth/logout', {
                method: 'POST',
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                console.log('✅ Logout bem-sucedido no backend');
            }
        } catch (error) {
            console.error('⚠️ Erro no logout do backend:', error);
        } finally {
            // Sempre limpar dados locais
            clearAuthData();
            updateUIForUnauthenticated();
            showNotification('Logout realizado com sucesso!', 'success');
            
            // Redirecionar se estiver em página protegida
            if (window.location.pathname.includes('dashboard') || 
                window.location.pathname.includes('history')) {
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            }
        }
    }

    // ==================== INICIALIZAÇÃO E SERVIÇOS ====================

    /**
     * Configura todos os event listeners
     */
    function setupEventListeners() {
        console.log('🔧 Configurando event listeners...');
        
        setupTradingEvents();
        setupRobotAI();
        setupChatbot();
        setupAccordionEvents();
    }

    /**
     * Configura eventos de acordeão
     */
    function setupAccordionEvents() {
        const accordionHeaders = document.querySelectorAll('.accordion-header');
        
        accordionHeaders.forEach(header => {
            header.addEventListener('click', function() {
                const content = this.nextElementSibling;
                this.classList.toggle('active');
                content.classList.toggle('open');
            });
        });
    }

    /**
     * Carrega dados iniciais
     */
    async function loadInitialData() {
        if (!currentUser) return;
        
        console.log('📥 Carregando dados iniciais...');
        
        await Promise.all([
            updateAccountBalance(),
            updateRobotStatus(),
            loadMarketAnalysis(),
            loadTradingSymbols()
        ]);
    }

    /**
     * Inicia serviços em background
     */
    function startBackgroundServices() {
        // Atualização periódica no dashboard
        if (window.location.pathname.includes('dashboard')) {
            setInterval(() => {
                if (currentUser) {
                    updateAccountBalance();
                    updateRobotStatus();
                }
            }, CONFIG.updateInterval);
        }
        
        // Verificação de saúde da conexão
        setInterval(() => {
            if (currentUser && !navigator.onLine) {
                showNotification('⚠️ Conexão perdida', 'warning');
            }
        }, 10000);
    }

    /**
     * Configura interface do usuário
     */
    function setupUserInterface() {
        console.log('🎨 Configurando interface...');
        
        // Inicializar componentes visuais
        updatePageSpecificUI();
        
        // Configurar temas/dark mode se necessário
        setupTheme();
    }

    /**
     * Configura tema da aplicação
     */
    function setupTheme() {
        // Implementação básica de tema
        const savedTheme = localStorage.getItem('financeclick_theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    // ==================== FUNÇÕES DE EXIBIÇÃO ====================

    /**
     * Exibe detalhes do contrato
     */
    function displayContractDetails(contract) {
        if (!elements.contractDetails) return;

        const statusClass = contract.status === 'win' ? 'status-win' : 
                          contract.status === 'loss' ? 'status-loss' : 'status-pending';

        elements.contractDetails.innerHTML = `
            <div class="contract-details-card">
                <h4>📋 Detalhes do Contrato</h4>
                <div class="contract-grid">
                    <div class="contract-item">
                        <label>ID do Contrato:</label>
                        <span class="contract-id">${contract.contract_id || 'N/A'}</span>
                    </div>
                    <div class="contract-item">
                        <label>Status:</label>
                        <span class="contract-status ${statusClass}">${contract.status || 'Aberto'}</span>
                    </div>
                    <div class="contract-item">
                        <label>Resultado:</label>
                        <span class="contract-result ${contract.result >= 0 ? 'positive' : 'negative'}">
                            $${contract.result || '0.00'}
                        </span>
                    </div>
                    <div class="contract-item">
                        <label>Ativo:</label>
                        <span class="contract-symbol">${contract.symbol || 'N/A'}</span>
                    </div>
                    <div class="contract-item">
                        <label>Taxa de Crescimento:</label>
                        <span class="contract-growth">${((contract.growth_rate || 0) * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Atualiza display do histórico
     */
    function updateHistoryDisplay(historyData) {
        // Implementar display do histórico
        console.log('📊 Atualizando display do histórico:', historyData);
    }

    // ==================== INICIALIZAÇÃO FINAL ====================

    /**
     * Função de inicialização global
     */
    function init() {
        console.log(`🚀 ${CONFIG.appName} v${CONFIG.version} - Iniciando...`);
        
        // Verificar suporte a APIs necessárias
        if (!window.localStorage) {
            showNotification('❌ Seu navegador não suporta localStorage', 'error');
            return;
        }
        
        if (!window.fetch) {
            showNotification('❌ Seu navegador é muito antigo', 'error');
            return;
        }
        
        // Inicializar aplicação
        initializeApplication();
        
        // Expor funções globais para debug
        window.financeClick = {
            version: CONFIG.version,
            getUser: () => currentUser,
            getAuthData: () => ({
                token: localStorage.getItem('deriv_token'),
                loginid: localStorage.getItem('deriv_loginid'),
                hasAuth: hasAuthData()
            }),
            clearAuth: clearAuthData,
            showNotification: showNotification
        };
        
        console.log('🎉 Aplicação carregada e pronta!');
    }

    // Inicializar quando o DOM estiver pronto
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // ==================== EXPORTAÇÕES GLOBAIS ====================
    
    window.handleLogin = handleLogin;
    window.handleLogout = handleLogout;
    window.showNotification = showNotification;
});