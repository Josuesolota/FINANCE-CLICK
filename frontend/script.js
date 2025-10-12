// script.js - VERSÃO FINAL CORRIGIDA - PROBLEMA OAUTH RESOLVIDO
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

    // ✅ CORREÇÃO CRÍTICA: Sistema de autenticação simplificado e robusto
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

    // ✅ CORREÇÃO: Sistema de estado simplificado
    function saveAuthData(loginid, token, currency = 'USD') {
        localStorage.setItem('deriv_token', token);
        localStorage.setItem('deriv_loginid', loginid);
        localStorage.setItem('deriv_currency', currency);
        console.log('💾 Dados de autenticação salvos:', loginid);
    }

    function clearAuthData() {
        localStorage.removeItem('deriv_token');
        localStorage.removeItem('deriv_loginid');
        localStorage.removeItem('deriv_currency');
        currentUser = null;
        console.log('🧹 Dados de autenticação limpos');
    }

    // ✅ CORREÇÃO: Verificar se estamos no callback OAuth
    function isOAuthCallback() {
        return window.location.search.includes('acct1=') || 
               window.location.search.includes('token1=');
    }

    // ✅ CORREÇÃO: Processar OAuth callback de forma robusta
    function processOAuthCallback() {
        console.log('🔄 Processando OAuth callback...');
        const urlParams = new URLSearchParams(window.location.search);
        
        let tokensProcessed = false;
        let i = 1;

        while (true) {
            const loginid = urlParams.get(`acct${i}`);
            const token = urlParams.get(`token${i}`);
            
            if (!loginid || !token) break;

            console.log(`📥 Encontrada conta ${i}:`, loginid);
            saveAuthData(loginid, token, urlParams.get(`cur${i}`) || 'USD');
            tokensProcessed = true;
            i++;
        }

        if (tokensProcessed) {
            console.log('✅ Tokens OAuth processados com sucesso');
            // Limpar URL parameters
            const cleanUrl = window.location.pathname;
            window.history.replaceState({}, document.title, cleanUrl);
            console.log('🧹 URL limpa:', cleanUrl);
        } else {
            console.log('ℹ️ Nenhum token OAuth encontrado na URL');
        }

        return tokensProcessed;
    }

    // ✅ CORREÇÃO: Inicialização otimizada
    async function initializeApp() {
        console.log('🚀 Inicializando aplicação...');
        highlightActiveLink();
        
        // Processar OAuth callback se estivermos em uma
        if (isOAuthCallback()) {
            console.log('🎯 Detectado OAuth callback - processando...');
            const hasTokens = processOAuthCallback();
            if (hasTokens) {
                // Restaurar sessão no backend imediatamente
                await restoreBackendSession();
            }
        }
        
        // Verificar autenticação atual
        await checkAuthentication();
        await loadInitialData();
        setupEventListeners();
        
        console.log('✅ Aplicação inicializada');
    }

    // ✅ CORREÇÃO: Sistema de autenticação otimizado
    async function checkAuthentication() {
        const token = localStorage.getItem('deriv_token');
        const loginid = localStorage.getItem('deriv_loginid');

        console.log('🔐 Verificando autenticação...', { 
            hasToken: !!token, 
            hasLoginId: !!loginid 
        });

        if (!token || !loginid) {
            console.log('❌ Dados de autenticação incompletos no localStorage');
            updateUINotAuthenticated();
            return;
        }

        try {
            const response = await fetch('/api/me', {
                headers: getAuthHeaders()
            });
            
            if (response.ok) {
                const userData = await response.json();
                console.log('✅ Autenticação válida:', userData.loginid);
                currentUser = userData;
                updateUIAuthenticated(userData);
            } else {
                console.log('❌ Autenticação inválida, status:', response.status);
                
                // Tentar restaurar sessão se for erro 401
                if (response.status === 401) {
                    console.log('🔄 Tentando restaurar sessão no backend...');
                    const restored = await restoreBackendSession();
                    if (restored) {
                        // Tentar autenticação novamente após restaurar
                        await checkAuthentication();
                        return;
                    }
                }
                
                // Se não conseguimos restaurar, limpar dados inválidos
                clearAuthData();
                updateUINotAuthenticated();
            }
        } catch (error) {
            console.error('❌ Erro de rede ao verificar autenticação:', error);
            // Em caso de erro de rede, manter UI atual
            updateUINotAuthenticated();
        }
    }

    // ✅ CORREÇÃO: Restaurar sessão no backend
    async function restoreBackendSession() {
        try {
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

    // ✅ CORREÇÃO: Atualização de UI mais confiável
    function updateUIAuthenticated(userData) {
        console.log('🎨 Atualizando UI para usuário autenticado:', userData.loginid);
        
        // Atualizar botão login/logout
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
            loginLogoutBtn.onclick = handleLogout;
        }

        // Atualizar informações do usuário
        if (userInfoElement) {
            userInfoElement.innerHTML = `
                <div class="user-welcome">
                    <span>Bem-vindo, ${userData.name || 'Trader'}!</span>
                    <small>Conta: ${userData.loginid}</small>
                </div>
            `;
            userInfoElement.style.display = 'block';
        }

        // Mostrar seções protegidas
        document.querySelectorAll('.protected-section').forEach(section => {
            section.style.display = 'block';
        });

        // Esconder seções de não autenticado
        document.querySelectorAll('.unauthenticated-section').forEach(section => {
            section.style.display = 'none';
        });

        console.log('✅ UI atualizada para autenticado');
    }

    function updateUINotAuthenticated() {
        console.log('🎨 Atualizando UI para não autenticado');
        
        // Atualizar botão login/logout
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
            loginLogoutBtn.onclick = handleLogin;
        }

        // Limpar informações do usuário
        if (userInfoElement) {
            userInfoElement.innerHTML = '';
            userInfoElement.style.display = 'none';
        }

        // Esconder seções protegidas
        document.querySelectorAll('.protected-section').forEach(section => {
            section.style.display = 'none';
        });

        // Mostrar seções de não autenticado
        document.querySelectorAll('.unauthenticated-section').forEach(section => {
            section.style.display = 'block';
        });

        console.log('✅ UI atualizada para não autenticado');
    }

    function handleLogin() {
        console.log('🔐 Redirecionando para OAuth...');
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
            
            // Redirecionar para página inicial se estiver em página protegida
            const currentPage = window.location.pathname;
            if (currentPage.includes('dashboard') || currentPage.includes('history')) {
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            }
        }
    }

    // 5. Sistema de Robô AI
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
            toggleRobotBtn.className = 'btn btn-danger';
            aiStatus.innerHTML = '<i class="fas fa-circle pulse"></i> Ligado';
            aiStatus.className = 'status active';
        } else {
            toggleRobotBtn.textContent = 'LIGAR ROBÔ';
            toggleRobotBtn.className = 'btn btn-primary';
            aiStatus.innerHTML = '<i class="fas fa-circle"></i> Desligado';
            aiStatus.className = 'status';
        }
    }

    // 6. Sistema de Trading
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

    // 8. Sistema de Notificações
    function showNotification(message, type = 'info') {
        // Remover notificações existentes
        document.querySelectorAll('.notification').forEach(n => n.remove());

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
            font-family: Arial, sans-serif;
            font-size: 14px;
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

        detailsElement.innerHTML = `
            <div class="contract-card">
                <h4>Detalhes do Contrato</h4>
                <div class="contract-info">
                    <p><strong>ID:</strong> ${contract.contract_id || 'N/A'}</p>
                    <p><strong>Status:</strong> <span class="status-${contract.status || 'open'}">${contract.status || 'Aberto'}</span></p>
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