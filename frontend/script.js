// script.js - VERSÃO FINAL - FLUXO OAUTH COMPLETAMENTE CORRIGIDO
document.addEventListener('DOMContentLoaded', () => {
    // 1. Variáveis Globais
    let currentUser = null;
    let isRobotActive = false;

    // Elementos DOM
    const menuToggle = document.getElementById('menuToggle');
    const mainNav = document.getElementById('mainNav');
    const loginLogoutBtn = document.getElementById('loginLogoutBtn');
    const userInfoElement = document.getElementById('userInfo');

    // 2. Inicialização da Aplicação
    initializeApp();

    // Função para destacar link ativo
    function highlightActiveLink() {
        if (!mainNav) return;

        const currentPath = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = mainNav.querySelectorAll('a.nav-link');

        navLinks.forEach(link => link.classList.remove('active'));
        
        navLinks.forEach(link => {
            const linkPath = link.getAttribute('href').split('/').pop();
            if (linkPath === currentPath) {
                link.classList.add('active');
            }
        });
    }

    // 3. Menu Hamburguer
    if (menuToggle && mainNav) {
        menuToggle.addEventListener('click', () => {
            mainNav.classList.toggle('open');
        });

        const navLinks = mainNav.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth < 768) {
                    mainNav.classList.remove('open');
                }
            });
        });
    }

    // ✅ CORREÇÃO CRÍTICA: Sistema de autenticação completamente refeito
    function getAuthHeaders() {
        const token = localStorage.getItem('deriv_token');
        const loginid = localStorage.getItem('deriv_loginid');
        
        if (token && loginid) {
            return {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
                'X-LoginID': loginid
            };
        }
        return {
            'Content-Type': 'application/json'
        };
    }

    // ✅ CORREÇÃO: Funções de estado simplificadas
    function saveAuthData(loginid, token, currency = 'USD') {
        localStorage.setItem('deriv_token', token);
        localStorage.setItem('deriv_loginid', loginid);
        localStorage.setItem('deriv_currency', currency);
        console.log('💾 Tokens salvos no localStorage:', loginid);
    }

    function clearAuthData() {
        localStorage.removeItem('deriv_token');
        localStorage.removeItem('deriv_loginid');
        localStorage.removeItem('deriv_currency');
        currentUser = null;
        console.log('🧹 Tokens removidos do localStorage');
    }

    function hasAuthData() {
        const token = localStorage.getItem('deriv_token');
        const loginid = localStorage.getItem('deriv_loginid');
        return !!(token && loginid);
    }

    // ✅ CORREÇÃO: Detecção e processamento de OAuth callback
    function processOAuthCallback() {
        console.log('🔄 Verificando parâmetros OAuth na URL...');
        
        const urlParams = new URLSearchParams(window.location.search);
        const urlString = window.location.search;
        
        console.log('📋 URL atual:', urlString);
        
        // Verificar se temos parâmetros OAuth
        if (urlParams.has('acct1') || urlParams.has('token1')) {
            console.log('🎯 Parâmetros OAuth detectados na URL!');
            
            let tokensFound = false;
            let i = 1;
            
            while (urlParams.has(`acct${i}`) && urlParams.has(`token${i}`)) {
                const loginid = urlParams.get(`acct${i}`);
                const token = urlParams.get(`token${i}`);
                const currency = urlParams.get(`cur${i}`) || 'USD';
                
                if (loginid && token) {
                    console.log(`📥 Processando conta ${i}:`, loginid);
                    saveAuthData(loginid, token, currency);
                    tokensFound = true;
                    
                    // Parar após a primeira conta válida
                    break;
                }
                i++;
            }
            
            if (tokensFound) {
                console.log('✅ Tokens OAuth processados com sucesso!');
                // Limpar URL - IMPORTANTE!
                const cleanUrl = window.location.pathname;
                window.history.replaceState({}, document.title, cleanUrl);
                console.log('🧹 URL limpa para:', cleanUrl);
                return true;
            }
        }
        
        console.log('ℹ️ Nenhum parâmetro OAuth encontrado na URL');
        return false;
    }

    // ✅ CORREÇÃO: Inicialização completamente refeita
    async function initializeApp() {
        console.log('🚀 Inicializando FinanceClick...');
        highlightActiveLink();
        
        // 1. Primeiro: processar OAuth callback se existir
        const hadOAuthCallback = processOAuthCallback();
        
        // 2. Verificar se temos dados de autenticação
        if (hasAuthData()) {
            console.log('🔐 Dados de autenticação encontrados no localStorage');
            await checkAuthentication();
        } else {
            console.log('🔐 Nenhum dado de autenticação encontrado');
            updateUINotAuthenticated();
        }
        
        // 3. Se tivemos um callback OAuth, restaurar sessão no backend
        if (hadOAuthCallback) {
            console.log('🔄 Restaurando sessão no backend após OAuth...');
            await restoreBackendSession();
            // Re-verificar autenticação após restaurar sessão
            await checkAuthentication();
        }
        
        // 4. Carregar dados e configurar eventos
        await loadInitialData();
        setupEventListeners();
        
        console.log('✅ Aplicação inicializada com sucesso!');
    }

    // ✅ CORREÇÃO: Sistema de autenticação simplificado
    async function checkAuthentication() {
        console.log('🔐 Verificando autenticação no backend...');
        
        if (!hasAuthData()) {
            console.log('❌ Sem dados de autenticação no localStorage');
            updateUINotAuthenticated();
            return;
        }

        try {
            const response = await fetch('/api/me', {
                headers: getAuthHeaders()
            });
            
            console.log('📡 Resposta do /api/me:', response.status);
            
            if (response.ok) {
                const userData = await response.json();
                console.log('✅ Autenticação VÁLIDA:', userData.loginid);
                currentUser = userData;
                updateUIAuthenticated(userData);
            } else {
                console.log('❌ Autenticação INVÁLIDA, status:', response.status);
                
                // Tentar restaurar sessão se for erro 401
                if (response.status === 401) {
                    console.log('🔄 Tentando restaurar sessão...');
                    const restored = await restoreBackendSession();
                    if (restored) {
                        // Tentar novamente após restaurar
                        await checkAuthentication();
                        return;
                    }
                }
                
                // Se não conseguiu restaurar, limpar dados inválidos
                console.log('🧹 Limpando dados de autenticação inválidos...');
                clearAuthData();
                updateUINotAuthenticated();
            }
        } catch (error) {
            console.error('💥 Erro ao verificar autenticação:', error);
            updateUINotAuthenticated();
        }
    }

    // ✅ CORREÇÃO: Restaurar sessão no backend
    async function restoreBackendSession() {
        try {
            console.log('🔄 Enviando tokens para o backend...');
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                console.log('✅ Sessão restaurada no backend!');
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

    // ✅ CORREÇÃO: Interface de usuário melhorada
    function updateUIAuthenticated(userData) {
        console.log('🎨 Atualizando UI para usuário autenticado...');
        
        // Botão login/logout
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
            loginLogoutBtn.onclick = handleLogout;
            loginLogoutBtn.style.display = 'block';
        }

        // Informações do usuário
        if (userInfoElement) {
            userInfoElement.innerHTML = `
                <div class="user-welcome" style="display: flex; flex-direction: column; align-items: flex-end;">
                    <span style="font-weight: bold;">Bem-vindo, ${userData.name || 'Trader'}!</span>
                    <small style="color: #666; font-size: 0.8em;">Conta: ${userData.loginid}</small>
                </div>
            `;
            userInfoElement.style.display = 'block';
        }

        // Mostrar conteúdo protegido
        document.querySelectorAll('.protected-content').forEach(el => {
            el.style.display = 'block';
        });

        // Esconder conteúdo para não autenticados
        document.querySelectorAll('.unauthenticated-content').forEach(el => {
            el.style.display = 'none';
        });

        console.log('✅ UI atualizada para modo autenticado');
    }

    function updateUINotAuthenticated() {
        console.log('🎨 Atualizando UI para não autenticado...');
        
        // Botão login/logout
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
            loginLogoutBtn.onclick = handleLogin;
            loginLogoutBtn.style.display = 'block';
        }

        // Informações do usuário
        if (userInfoElement) {
            userInfoElement.innerHTML = '';
            userInfoElement.style.display = 'none';
        }

        // Esconder conteúdo protegido
        document.querySelectorAll('.protected-content').forEach(el => {
            el.style.display = 'none';
        });

        // Mostrar conteúdo para não autenticados
        document.querySelectorAll('.unauthenticated-content').forEach(el => {
            el.style.display = 'block';
        });

        console.log('✅ UI atualizada para modo não autenticado');
    }

    // ✅ CORREÇÃO: Handlers de login/logout
    function handleLogin() {
        console.log('🔐 Iniciando processo de login OAuth...');
        console.log('📍 Redirecionando para: /auth/login');
        window.location.href = '/auth/login';
    }

    async function handleLogout() {
        console.log('👋 Iniciando logout...');
        
        try {
            const response = await fetch('/auth/logout', { 
                method: 'POST',
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                console.log('✅ Logout bem-sucedido no backend');
            } else {
                console.log('⚠️ Logout no backend falhou, mas continuando...');
            }
        } catch (error) {
            console.error('⚠️ Erro no logout do backend:', error);
        } finally {
            // SEMPRE limpar dados locais
            clearAuthData();
            updateUINotAuthenticated();
            showNotification('Logout realizado com sucesso!', 'success');
            
            // Redirecionar se estiver em página protegida
            const currentPath = window.location.pathname;
            if (currentPath.includes('dashboard') || currentPath.includes('history')) {
                console.log('🔄 Redirecionando para página inicial...');
                setTimeout(() => {
                    window.location.href = '/';
                }, 1000);
            }
        }
    }

    // 5. Sistema de Robô AI (mantido com headers)
    function setupRobotAIControls() {
        const toggleRobotBtn = document.getElementById('toggleRobotBtn');
        const aiStatus = document.getElementById('aiStatus');

        if (!toggleRobotBtn || !aiStatus) return;

        checkRobotStatus();

        toggleRobotBtn.addEventListener('click', async () => {
            if (!currentUser) {
                showNotification('Por favor, faça login primeiro!', 'warning');
                return;
            }

            const config = {
                strategy: document.getElementById('strategySelect')?.value || 'moderate',
                trade_amount: parseFloat(document.getElementById('tradeAmount')?.value) || 5,
                growth_rate: 0.02
            };

            try {
                const response = await fetch('/api/robot/toggle', {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify(config)
                });

                if (response.ok) {
                    const result = await response.json();
                    isRobotActive = result.status === 'running';
                    updateRobotUI(isRobotActive);
                    showNotification(
                        isRobotActive ? '🤖 Robô AI ativado!' : '🤖 Robô AI desativado',
                        isRobotActive ? 'success' : 'info'
                    );
                }
            } catch (error) {
                console.error('Erro ao alternar robô:', error);
                showNotification('Erro de comunicação com o servidor', 'error');
            }
        });
    }

    async function checkRobotStatus() {
        try {
            const response = await fetch('/api/robot/status', {
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                const status = await response.json();
                isRobotActive = status.active;
                updateRobotUI(isRobotActive);
            }
        } catch (error) {
            console.error('Erro ao verificar status do robô:', error);
        }
    }

    function updateRobotUI(isActive) {
        const toggleRobotBtn = document.getElementById('toggleRobotBtn');
        const aiStatus = document.getElementById('aiStatus');

        if (!toggleRobotBtn || !aiStatus) return;

        if (isActive) {
            toggleRobotBtn.textContent = '🛑 DESLIGAR ROBÔ';
            toggleRobotBtn.className = 'btn btn-danger';
            aiStatus.innerHTML = '<span style="color: #4CAF50;">●</span> ATIVO';
            aiStatus.className = 'status active';
        } else {
            toggleRobotBtn.textContent = '🚀 LIGAR ROBÔ';
            toggleRobotBtn.className = 'btn btn-primary';
            aiStatus.innerHTML = '<span style="color: #666;">●</span> INATIVO';
            aiStatus.className = 'status';
        }
    }

    // 6. Sistema de Trading (mantido com headers)
    function setupAccumulatorTrading() {
        const buyButton = document.getElementById('buyAccumulatorBtn');
        if (!buyButton) return;

        loadAccumulatorSymbols();

        buyButton.addEventListener('click', async () => {
            if (!currentUser) {
                showNotification('Por favor, faça login primeiro!', 'warning');
                return;
            }

            const tradeData = {
                symbol: document.getElementById('symbolSelect')?.value || '1HZ100V',
                growth_rate: parseFloat(document.getElementById('growthRate')?.value) || 0.02,
                amount: parseFloat(document.getElementById('amount')?.value) || 5,
                duration: 60
            };

            try {
                showNotification('🔄 Executando compra...', 'info');
                
                const response = await fetch('/api/accumulators/buy', {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify(tradeData)
                });

                if (response.ok) {
                    const result = await response.json();
                    showNotification('✅ Compra executada com sucesso!', 'success');
                    await updateAccountBalance();
                    
                    if (result.buy) {
                        displayContractDetails(result.buy);
                    }
                } else {
                    showNotification('❌ Erro na compra', 'error');
                }
            } catch (error) {
                console.error('Erro na compra:', error);
                showNotification('💥 Erro de comunicação', 'error');
            }
        });
    }

    async function loadAccumulatorSymbols() {
        const symbolSelect = document.getElementById('symbolSelect');
        if (!symbolSelect) return;

        try {
            const response = await fetch('/api/symbols/accumulators', {
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                const data = await response.json();
                symbolSelect.innerHTML = '';
                
                data.accumulator_symbols.forEach(symbol => {
                    const option = document.createElement('option');
                    option.value = symbol.symbol;
                    option.textContent = `${symbol.display_name} (${symbol.symbol})`;
                    symbolSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Erro ao carregar símbolos:', error);
        }
    }

    // 7. Sistema de Saldo
    async function updateAccountBalance() {
        const balanceElement = document.getElementById('accountBalance');
        if (!balanceElement) return;

        try {
            const response = await fetch('/api/balance', {
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.balance) {
                    balanceElement.textContent = 
                        `$${data.balance.balance.toFixed(2)} ${data.balance.currency || 'USD'}`;
                }
            }
        } catch (error) {
            console.error('Erro ao atualizar saldo:', error);
            balanceElement.textContent = 'Erro ao carregar';
        }
    }

    // 8. Sistema de Notificações (melhorado)
    function showNotification(message, type = 'info') {
        // Remover notificações existentes
        document.querySelectorAll('.notification').forEach(n => n.remove());

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()" style="background: none; border: none; color: white; font-size: 18px; cursor: pointer; margin-left: 10px;">×</button>
        `;

        const styles = {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '15px 20px',
            borderRadius: '5px',
            color: 'white',
            zIndex: '10000',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            minWidth: '300px',
            animation: 'slideIn 0.3s ease',
            fontFamily: 'Arial, sans-serif',
            fontSize: '14px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
        };

        Object.assign(notification.style, styles);

        const colors = {
            success: '#4CAF50',
            error: '#f44336',
            warning: '#ff9800',
            info: '#2196F3'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(notification);

        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    // 9. Sistema do Chatbot
    function setupChatbot() {
        const chatInput = document.getElementById('chatInput');
        const sendChatBtn = document.getElementById('sendChatBtn');
        const chatMessages = document.getElementById('chatMessages');

        if (!chatInput || !sendChatBtn || !chatMessages) return;

        function addMessage(message, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = message;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendMessage() {
            const query = chatInput.value.trim();
            if (!query) return;

            addMessage(query, true);
            chatInput.value = '';

            try {
                const response = await fetch('/api/chatbot/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query })
                });

                if (response.ok) {
                    const data = await response.json();
                    addMessage(data.response);
                } else {
                    addMessage('Desculpe, houve um erro ao processar sua pergunta.');
                }
            } catch (error) {
                console.error('Erro no chatbot:', error);
                addMessage('Desculpe, estou com problemas de conexão.');
            }
        }

        sendChatBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

    // 10. Funções Auxiliares
    function displayContractDetails(contract) {
        const detailsElement = document.getElementById('contractDetails');
        if (!detailsElement) return;

        const statusClass = contract.status === 'win' ? 'status-win' : contract.status === 'loss' ? 'status-loss' : 'status-open';
        
        detailsElement.innerHTML = `
            <div class="contract-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background: #f9f9f9;">
                <h4 style="margin: 0 0 10px 0;">📋 Detalhes do Contrato</h4>
                <div class="contract-info">
                    <p><strong>ID:</strong> ${contract.contract_id || 'N/A'}</p>
                    <p><strong>Status:</strong> <span class="${statusClass}">${contract.status || 'Aberto'}</span></p>
                    <p><strong>Resultado:</strong> $${contract.result || '0.00'}</p>
                    <p><strong>Symbol:</strong> ${contract.symbol || 'N/A'}</p>
                </div>
            </div>
        `;
    }

    async function loadInitialData() {
        if (currentUser) {
            await updateAccountBalance();
            await checkRobotStatus();
        }
    }

    function setupEventListeners() {
        setupRobotAIControls();
        setupAccumulatorTrading();
        setupChatbot();

        // Atualizar dados periodicamente no dashboard
        if (window.location.pathname.includes('dashboard')) {
            setInterval(async () => {
                if (currentUser) {
                    await updateAccountBalance();
                    await checkRobotStatus();
                }
            }, 30000);
        }
    }

    // 11. Acordeão
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    accordionHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            header.classList.toggle('active');
            content.classList.toggle('open');
        });
    });

    // Expor funções globais
    window.showNotification = showNotification;
    window.handleLogin = handleLogin;
    window.handleLogout = handleLogout;

    // Adicionar estilos CSS para notificações se não existirem
    if (!document.querySelector('#notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            .status-win { color: #4CAF50; font-weight: bold; }
            .status-loss { color: #f44336; font-weight: bold; }
            .status-open { color: #ff9800; font-weight: bold; }
        `;
        document.head.appendChild(style);
    }
});