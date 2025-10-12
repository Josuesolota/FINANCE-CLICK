// script.js - VERSÃO CORRIGIDA - ESTADO PERSISTENTE ENTRE PÁGINAS
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

    // ✅ CORREÇÃO CRÍTICA: Sistema de estado persistente
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

    // ✅ CORREÇÃO: Processar OAuth callback em TODAS as páginas
    function handleOAuthCallback() {
        const urlParams = new URLSearchParams(window.location.search);
        
        console.log('🔄 Verificando callback OAuth...');
        
        let tokensFound = false;
        let i = 1;
        
        while (urlParams.get(`acct${i}`)) {
            const loginid = urlParams.get(`acct${i}`);
            const token = urlParams.get(`token${i}`);
            
            if (loginid && token) {
                localStorage.setItem('deriv_token', token);
                localStorage.setItem('deriv_loginid', loginid);
                localStorage.setItem('deriv_currency', urlParams.get(`cur${i}`) || 'USD');
                
                console.log('✅ Tokens salvos no localStorage:', loginid);
                tokensFound = true;
                
                // Limpar URL parameters
                if (window.location.search.includes('acct1')) {
                    const newUrl = window.location.pathname;
                    window.history.replaceState({}, document.title, newUrl);
                    console.log('🧹 Parâmetros da URL limpos');
                }
                break;
            }
            i++;
        }
        
        return tokensFound;
    }

    function clearAuthData() {
        localStorage.removeItem('deriv_token');
        localStorage.removeItem('deriv_loginid');
        localStorage.removeItem('deriv_currency');
        currentUser = null;
        console.log('🧹 Dados de autenticação limpos');
    }

    // ✅ CORREÇÃO: Função de inicialização robusta
    async function initializeApp() {
        console.log('🚀 Inicializando aplicação...');
        highlightActiveLink();
        
        // Processar OAuth callback primeiro (se houver)
        const hasNewTokens = handleOAuthCallback();
        
        // Verificar autenticação
        await checkAuthentication();
        
        // Se temos novos tokens do OAuth, restaurar sessão no backend
        if (hasNewTokens) {
            await restoreBackendSession();
        }
        
        await loadInitialData();
        setupEventListeners();
        
        console.log('✅ Aplicação inicializada');
    }

    // ✅ CORREÇÃO: Sistema de autenticação melhorado
    async function checkAuthentication() {
        const token = localStorage.getItem('deriv_token');
        const loginid = localStorage.getItem('deriv_loginid');

        console.log('🔐 Verificando autenticação...', { 
            token: token ? 'PRESENTE' : 'AUSENTE', 
            loginid: loginid || 'AUSENTE' 
        });

        if (!token || !loginid) {
            console.log('❌ Sem tokens no localStorage');
            updateUINotAuthenticated();
            return;
        }

        try {
            console.log('🔄 Consultando /api/me...');
            const response = await fetch('/api/me', {
                headers: getAuthHeaders()
            });
            
            console.log('📥 Status da resposta:', response.status);
            
            if (response.ok) {
                const userData = await response.json();
                console.log('✅ Usuário autenticado:', userData.loginid);
                currentUser = userData;
                updateUIAuthenticated(userData);
            } else {
                console.log('❌ Falha na autenticação, status:', response.status);
                
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
                
                clearAuthData();
                updateUINotAuthenticated();
            }
        } catch (error) {
            console.error('❌ Erro ao verificar autenticação:', error);
            // Não limpar dados em caso de erro de rede - manter estado atual
            updateUINotAuthenticated();
        }
    }

    // ✅ CORREÇÃO: Restaurar sessão no backend
    async function restoreBackendSession() {
        try {
            console.log('🔄 Restaurando sessão no backend...');
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                console.log('✅ Sessão restaurada no backend');
                return true;
            } else {
                console.log('❌ Falha ao restaurar sessão:', response.status);
                return false;
            }
        } catch (error) {
            console.error('❌ Erro ao restaurar sessão:', error);
            return false;
        }
    }

    // ✅ CORREÇÃO: Atualização de UI mais robusta
    function updateUIAuthenticated(userData) {
        console.log('🎨 Atualizando UI para usuário autenticado:', userData.loginid);
        
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
            loginLogoutBtn.onclick = handleLogout;
            console.log('✅ Botão logout configurado');
        }

        if (userInfoElement) {
            userInfoElement.innerHTML = `
                <div class="user-welcome">
                    <span>Bem-vindo, ${userData.name || 'Trader'}!</span>
                    <small>Conta: ${userData.loginid}</small>
                </div>
            `;
            console.log('✅ Informações do usuário exibidas');
        }

        // Atualizar saldo se estiver no dashboard
        if (document.getElementById('accountBalance')) {
            updateAccountBalance();
        }

        // Mostrar seções protegidas
        document.querySelectorAll('.protected-section').forEach(section => {
            section.style.display = 'block';
        });
    }

    function updateUINotAuthenticated() {
        console.log('🎨 Atualizando UI para não autenticado');
        
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
            loginLogoutBtn.onclick = handleLogin;
        }

        if (userInfoElement) {
            userInfoElement.innerHTML = '';
        }

        // Esconder seções protegidas
        document.querySelectorAll('.protected-section').forEach(section => {
            section.style.display = 'none';
        });
    }

    function handleLogin() {
        console.log('🔐 Redirecionando para login OAuth...');
        window.location.href = '/auth/login';
    }

    async function handleLogout() {
        try {
            console.log('👋 Executando logout...');
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
            // SEMPRE limpar dados locais, mesmo se o backend falhar
            clearAuthData();
            updateUINotAuthenticated();
            showNotification('Logout realizado com sucesso!', 'success');
            
            // Redirecionar se estiver em página protegida
            if (window.location.pathname.includes('dashboard') || 
                window.location.pathname.includes('history')) {
                setTimeout(() => window.location.href = '/', 1000);
            }
        }
    }

    // 5. Sistema de Robô AI (mantido igual, mas com headers)
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
                        isRobotActive ? 'Robô AI ativado!' : 'Robô AI desativado',
                        isRobotActive ? 'success' : 'info'
                    );
                }
            } catch (error) {
                console.error('Erro ao alternar robô:', error);
                showNotification('Erro de comunicação', 'error');
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
            toggleRobotBtn.textContent = 'DESLIGAR ROBÔ';
            toggleRobotBtn.className = 'btn btn-secondary';
            aiStatus.innerHTML = '<i class="fas fa-circle pulse"></i> Ligado';
            aiStatus.className = 'status active';
        } else {
            toggleRobotBtn.textContent = 'LIGAR ROBÔ';
            toggleRobotBtn.className = 'btn btn-primary';
            aiStatus.innerHTML = '<i class="fas fa-circle"></i> Desligado';
            aiStatus.className = 'status';
        }
    }

    // 6. Sistema de Trading (mantido igual, mas com headers)
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
                showNotification('Executando compra...', 'info');
                
                const response = await fetch('/api/accumulators/buy', {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: JSON.stringify(tradeData)
                });

                if (response.ok) {
                    const result = await response.json();
                    showNotification('Compra executada com sucesso!', 'success');
                    await updateAccountBalance();
                    
                    if (result.buy) {
                        displayContractDetails(result.buy);
                    }
                } else {
                    showNotification('Erro na compra', 'error');
                }
            } catch (error) {
                console.error('Erro na compra:', error);
                showNotification('Erro de comunicação', 'error');
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

    // 7. Sistema de Saldo (com headers)
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

    // 8. Sistema de Notificações (mantido igual)
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">&times;</button>
        `;

        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            z-index: 10000;
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-width: 300px;
            animation: slideIn 0.3s ease;
        `;

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

    // 9. Sistema do Chatbot (mantido igual)
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

        detailsElement.innerHTML = `
            <div class="contract-card">
                <h4>Detalhes do Contrato</h4>
                <div class="contract-info">
                    <p><strong>ID:</strong> ${contract.contract_id || 'N/A'}</p>
                    <p><strong>Status:</strong> ${contract.status || 'Aberto'}</p>
                    <p><strong>Resultado:</strong> $${contract.result || '0.00'}</p>
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
});