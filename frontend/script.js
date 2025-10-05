document.addEventListener('DOMContentLoaded', () => {
    // 1. Variáveis Globais e Estado da Aplicação
    let currentUser = null;
    let isRobotActive = false;
    let currentBalance = 0;

    // Elementos DOM
    const menuToggle = document.getElementById('menuToggle');
    const mainNav = document.getElementById('mainNav');
    const loginLogoutBtn = document.getElementById('loginLogoutBtn');
    const userInfoElement = document.getElementById('userInfo');

    // 2. Inicialização da Aplicação
    initializeApp();

    // ----------------------------------------------------
    // NOVO: LÓGICA PARA DESTACAR O LINK ATIVO (.active)
    // ----------------------------------------------------
    function highlightActiveLink() {
        if (!mainNav) return;

        const currentPath = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = mainNav.querySelectorAll('a.nav-link');

        // Remove a classe 'active' de todos os links primeiro
        navLinks.forEach(link => {
            link.classList.remove('active');
        });

        // Adiciona a classe 'active' ao link correspondente
        navLinks.forEach(link => {
            const linkPath = link.getAttribute('href').split('/').pop();
            
            // Compara o caminho do link com o nome do arquivo da página atual
            if (linkPath === currentPath) {
                link.classList.add('active');
            }
        });
    }
    // ----------------------------------------------------
    // FIM DA LÓGICA ACTIVE LINK
    // ----------------------------------------------------

    // 3. Toggle do Menu Hamburguer (mantido da versão original)
    if (menuToggle && mainNav) {
        menuToggle.addEventListener('click', () => {
            mainNav.classList.toggle('open');
        });

        const navLinks = mainNav.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                // Fecha o menu móvel ao clicar em um link, apenas em telas pequenas
                if (window.innerWidth < 768) {
                    mainNav.classList.remove('open');
                }
            });
        });
    }

    // 4. Sistema de Autenticação Real com Deriv API
    async function initializeApp() {
        highlightActiveLink(); // Chamada adicionada para rodar na inicialização
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

        // Atualizar saldo se estiver no dashboard
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
            await fetch('/auth/logout', { method: 'POST' });
            currentUser = null;
            updateUINotAuthenticated();
            showNotification('Logout realizado com sucesso!', 'success');
            
            // Redirecionar para página inicial se estiver no dashboard
            if (window.location.pathname.includes('dashboard')) {
                setTimeout(() => window.location.href = '/', 1000);
            }
        } catch (error) {
            console.error('Erro no logout:', error);
            showNotification('Erro ao fazer logout', 'error');
        }
    }

    // 5. Sistema de Robô AI para Accumulator Options
    function setupRobotAIControls() {
        const toggleRobotBtn = document.getElementById('toggleRobotBtn');
        const aiStatus = document.getElementById('aiStatus');
        const strategySelect = document.getElementById('strategySelect');
        const stopLossInput = document.getElementById('stopLoss');
        const takeProfitInput = document.getElementById('takeProfit');
        const tradeAmountInput = document.getElementById('tradeAmount');
        const tradeAmountValue = document.getElementById('tradeAmountValue');

        if (!toggleRobotBtn || !aiStatus) return;

        // Atualizar valor do trade amount
        if (tradeAmountInput && tradeAmountValue) {
            tradeAmountValue.textContent = `$${tradeAmountInput.value}.00`;
            tradeAmountInput.addEventListener('input', () => {
                tradeAmountValue.textContent = `$${tradeAmountInput.value}.00`;
            });
        }

        // Verificar status atual do robô
        checkRobotStatus();

        // Configurar toggle do robô
        toggleRobotBtn.addEventListener('click', async () => {
            if (!currentUser) {
                showNotification('Por favor, faça login primeiro!', 'warning');
                return;
            }

            const config = {
                strategy: strategySelect?.value || 'moderate',
                max_daily_loss: parseFloat(stopLossInput?.value) || 100,
                take_profit_ticks: parseInt(takeProfitInput?.value) || 10,
                stop_loss_ticks: parseInt(stopLossInput?.value) || 3,
                trade_amount: parseFloat(tradeAmountInput?.value) || 5,
                growth_rate: 0.02
            };

            try {
                const response = await fetch('/api/robot/toggle', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(config)
                });

                if (response.ok) {
                    const result = await response.json();
                    isRobotActive = result.status === 'running';
                    updateRobotUI(isRobotActive);
                    showNotification(
                        isRobotActive 
                            ? `Robô AI ativado com estratégia ${config.strategy}`
                            : 'Robô AI desativado',
                        isRobotActive ? 'success' : 'info'
                    );

                    // Mostrar análise de mercado se o robô foi ativado
                    if (isRobotActive && result.analysis) {
                        displayMarketAnalysis(result.analysis);
                    }
                } else {
                    const error = await response.json();
                    showNotification(`Erro: ${error.detail}`, 'error');
                }
            } catch (error) {
                console.error('Erro ao alternar robô:', error);
                showNotification('Erro de comunicação com o servidor', 'error');
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

    // 6. Sistema de Trading com Accumulator Options
    function setupAccumulatorTrading() {
        const symbolSelect = document.getElementById('symbolSelect');
        const growthRateSelect = document.getElementById('growthRate');
        const amountInput = document.getElementById('amount');
        const buyButton = document.getElementById('buyAccumulatorBtn');
        const proposalButton = document.getElementById('getProposalBtn');
        const proposalResult = document.getElementById('proposalResult');

        if (!buyButton) return;

        // Carregar símbolos disponíveis
        loadAccumulatorSymbols();

        // Configurar botão de proposta
        if (proposalButton && proposalResult) {
            proposalButton.addEventListener('click', async () => {
                const proposal = await getAccumulatorProposal();
                if (proposal) {
                    proposalResult.innerHTML = `
                        <div class="proposal-info">
                            <h4>Proposta Recebida</h4>
                            <p>Payout Potencial: $${proposal.proposal?.display_value || 'N/A'}</p>
                            <p>Taxa de Crescimento: ${(growthRateSelect.value * 100)}%</p>
                        </div>
                    `;
                }
            });
        }

        // Configurar botão de compra
        buyButton.addEventListener('click', async () => {
            if (!currentUser) {
                showNotification('Por favor, faça login primeiro!', 'warning');
                return;
            }

            const tradeData = {
                symbol: symbolSelect?.value || '1HZ100V',
                growth_rate: parseFloat(growthRateSelect?.value) || 0.02,
                amount: parseFloat(amountInput?.value) || 5,
                duration: 60,
                duration_unit: 't'
            };

            try {
                showNotification('Executando compra...', 'info');
                
                const response = await fetch('/api/accumulators/buy', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(tradeData)
                });

                const result = await response.json();

                if (response.ok) {
                    showNotification('Compra executada com sucesso!', 'success');
                    console.log('Resultado da compra:', result);
                    
                    // Atualizar saldo após compra
                    await updateAccountBalance();
                    
                    // Mostrar detalhes do contrato se disponível
                    if (result.buy) {
                        displayContractDetails(result.buy);
                    }
                } else {
                    showNotification(`Erro na compra: ${result.detail}`, 'error');
                }
            } catch (error) {
                console.error('Erro na compra:', error);
                showNotification('Erro de comunicação com o servidor', 'error');
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

    async function getAccumulatorProposal() {
        const symbolSelect = document.getElementById('symbolSelect');
        const growthRateSelect = document.getElementById('growthRate');
        const amountInput = document.getElementById('amount');

        const proposalData = {
            symbol: symbolSelect?.value || '1HZ100V',
            growth_rate: parseFloat(growthRateSelect?.value) || 0.02,
            amount: parseFloat(amountInput?.value) || 5,
            duration: 60,
            duration_unit: 't'
        };

        try {
            const response = await fetch('/api/accumulators/proposal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(proposalData)
            });

            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Erro ao obter proposta:', error);
        }
        return null;
    }

    // 7. Sistema de Saldo e Dados da Conta
    async function updateAccountBalance() {
        const balanceElement = document.getElementById('accountBalance');
        if (!balanceElement) return;

        try {
            const response = await fetch('/api/balance');
            if (response.ok) {
                const data = await response.json();
                if (data.balance) {
                    currentBalance = data.balance.balance;
                    balanceElement.textContent = 
                        `$${currentBalance.toFixed(2)} ${data.balance.currency || 'USD'}`;
                }
            }
        } catch (error) {
            console.error('Erro ao atualizar saldo:', error);
            balanceElement.textContent = 'Erro ao carregar';
        }
    }

    // 8. Sistema de Análise de Mercado
    async function loadMarketAnalysis() {
        const analysisElement = document.getElementById('marketAnalysis');
        if (!analysisElement) return;

        try {
            const response = await fetch('/api/market/analysis?symbol=1HZ100V&strategy=moderate');
            if (response.ok) {
                const analysis = await response.json();
                displayMarketAnalysis(analysis);
            }
        } catch (error) {
            console.error('Erro ao carregar análise:', error);
        }
    }

    function displayMarketAnalysis(analysis) {
        const analysisElement = document.getElementById('marketAnalysis');
        if (!analysisElement) return;

        analysisElement.innerHTML = `
            <div class="analysis-card">
                <h4>Análise do Mercado</h4>
                <div class="analysis-metrics">
                    <div class="metric">
                        <label>Volatilidade:</label>
                        <span class="value ${analysis.volatility > 0.7 ? 'high' : analysis.volatility > 0.4 ? 'medium' : 'low'}">
                            ${(analysis.volatility * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="metric">
                        <label>Probabilidade de Sucesso:</label>
                        <span class="value ${analysis.success_probability > 0.7 ? 'high' : analysis.success_probability > 0.4 ? 'medium' : 'low'}">
                            ${(analysis.success_probability * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="metric">
                        <label>Taxa Recomendada:</label>
                        <span class="value">${(analysis.recommended_growth_rate * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    // 9. Sistema de Notificações (mantido)
    function showNotification(message, type = 'info') {
        // ... (código showNotification) ...
    }

    // 10. Sistema do Chatbot (mantido)
    function setupChatbot() {
        // ... (código setupChatbot) ...
    }

    // 11. Funções de Utilidade
    function displayContractDetails(contract) {
        const detailsElement = document.getElementById('contractDetails');
        if (!detailsElement) return;

        detailsElement.innerHTML = `
            <div class="contract-card">
                <h4>Detalhes do Contrato</h4>
                <div class="contract-info">
                    <p><strong>ID:</strong> ${contract.contract_id || 'N/A'}</p>
                    <p><strong>Status:</strong> ${contract.status || 'Aberto'}</p>
                    <p><strong>Valor:</strong> $${contract.amount || 'N/A'}</p>
                    <p><strong>Symbol:</strong> ${contract.symbol || 'N/A'}</p>
                </div>
            </div>
        `;
    }

    async function loadInitialData() {
        if (currentUser) {
            await updateAccountBalance();
            await checkRobotStatus();
            await loadMarketAnalysis();
        }
    }

    function setupEventListeners() {
        setupRobotAIControls();
        setupAccumulatorTrading();
        setupChatbot();

        // Atualizar dados a cada 30 segundos se estiver no dashboard
        if (window.location.pathname.includes('dashboard')) {
            setInterval(async () => {
                if (currentUser) {
                    await updateAccountBalance();
                    await checkRobotStatus();
                }
            }, 30000);
        }
    }

    // 12. Acordeão (mantido)
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    
    if (accordionHeaders.length > 0) {
        accordionHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const content = header.nextElementSibling;
                
                accordionHeaders.forEach(h => {
                    if (h !== header) {
                        h.nextElementSibling.classList.remove('open');
                    }
                });

                content.classList.toggle('open');
            });
        });
    }

    // Adicionar estilos CSS para componentes dinâmicos (mantido)
    const dynamicStyles = `
        <style>
            /* ... (Estilos CSS dinâmicos) ... */
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 5px;
                color: white;
                z-index: 1000;
                display: flex;
                justify-content: space-between;
                align-items: center;
                min-width: 300px;
                max-width: 400px;
                animation: slideIn 0.3s ease;
            }
            
            .notification.success { background: #4CAF50; }
            .notification.error { background: #f44336; }
            .notification.warning { background: #ff9800; }
            .notification.info { background: #2196F3; }
            
            .notification button {
                background: none;
                border: none;
                color: white;
                font-size: 18px;
                cursor: pointer;
                margin-left: 10px;
            }
            
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .status.active { color: #4CAF50; }
            .status { color: #666; }
            
            .pulse {
                animation: pulse 1.5s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .analysis-card, .contract-card, .proposal-info {
                background: #1a1a1a; /* Ajustado para melhor integração com Dark Mode */
                color: #FFFFFF;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
            
            .metric {
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
            }
            
            .value.high { color: #4CAF50; }
            .value.medium { color: #ff9800; }
            .value.low { color: #f44336; }
            
            .chat-message {
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
            }
            
            .user-message {
                background: #2a2a2a; /* Ajustado para Dark Mode */
                color: #FFFFFF;
                text-align: right;
                margin-left: auto;
            }
            
            .bot-message {
                background: #3a3a3a; /* Ajustado para Dark Mode */
                color: #FFFFFF;
            }
        </style>
    `;
    
    document.head.insertAdjacentHTML('beforeend', dynamicStyles);
});