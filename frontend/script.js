// script.js - VERSÃO CORRIGIDA E SIMPLIFICADA
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

    // 4. Sistema de Autenticação
    async function initializeApp() {
        highlightActiveLink();
        await checkAuthentication();
        await loadInitialData();
        setupEventListeners();
    }

    async function checkAuthentication() {
        try {
            const response = await fetch('/api/me');
            
            if (response.ok) {
                const userData = await response.json();
                currentUser = userData;
                updateUIAuthenticated(userData);
            } else {
                updateUINotAuthenticated();
            }
        } catch (error) {
            console.error('Erro ao verificar autenticação:', error);
            updateUINotAuthenticated();
        }
    }

    function updateUIAuthenticated(userData) {
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
            loginLogoutBtn.onclick = handleLogout;
        }

        if (userInfoElement) {
            userInfoElement.innerHTML = `
                <div class="user-welcome">
                    <span>Bem-vindo, ${userData.name || 'Trader'}!</span>
                    <small>Conta: ${userData.loginid}</small>
                </div>
            `;
        }

        if (document.getElementById('accountBalance')) {
            updateAccountBalance();
        }
    }

    function updateUINotAuthenticated() {
        if (loginLogoutBtn) {
            loginLogoutBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
            loginLogoutBtn.onclick = handleLogin;
        }

        if (userInfoElement) {
            userInfoElement.innerHTML = '';
        }
    }

    function handleLogin() {
        window.location.href = '/auth/login';
    }

    async function handleLogout() {
        try {
            const response = await fetch('/auth/logout', { 
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            if (response.ok) {
                currentUser = null;
                updateUINotAuthenticated();
                showNotification('Logout realizado com sucesso!', 'success');
                
                if (window.location.pathname.includes('dashboard')) {
                    setTimeout(() => window.location.href = '/', 1000);
                }
            }
        } catch (error) {
            console.error('Erro no logout:', error);
            showNotification('Erro ao fazer logout', 'error');
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
                    headers: {'Content-Type': 'application/json'},
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
                } else {
                    showNotification('Erro ao controlar robô', 'error');
                }
            } catch (error) {
                console.error('Erro ao alternar robô:', error);
                showNotification('Erro de comunicação', 'error');
            }
        });
    }

    async function checkRobotStatus() {
        try {
            const response = await fetch('/api/robot/status');
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
                    headers: {'Content-Type': 'application/json'},
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
            const response = await fetch('/api/symbols/accumulators');
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
            const response = await fetch('/api/balance');
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