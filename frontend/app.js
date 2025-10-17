// MSP Intelligence Mesh Network - Shared JavaScript

const API_BASE_URL = 'https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod';

// Create global MSP namespace
window.MSP = window.MSP || {};

// Utility Functions
MSP.showLoading = (elementId) => {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="loading loading-lg"></div>';
    }
};

MSP.hideLoading = (elementId) => {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '';
    }
};

MSP.showAlert = (message, type = 'info') => {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} animate-slide-in`;
    alertDiv.innerHTML = `
        <span>${type === 'success' ? '✅' : type === 'danger' ? '❌' : 'ℹ️'}</span>
        <span>${message}</span>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        setTimeout(() => alertDiv.remove(), 5000);
    }
};

// API Functions
MSP.apiCall = async (endpoint, options = {}) => {
    try {
        const url = `${API_BASE_URL}${endpoint}`;
        console.log('API Call:', url);
        
        const response = await fetch(url, {
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('API Response:', data);
        return data;
    } catch (error) {
        console.error('API Call Error:', error);
        throw error;
    }
};

// Legacy support
const showLoading = MSP.showLoading;
const hideLoading = MSP.hideLoading;
const showAlert = MSP.showAlert;
const apiCall = MSP.apiCall;

// Dropdown Handler
document.addEventListener('DOMContentLoaded', () => {
    const dropdowns = document.querySelectorAll('.dropdown');
    
    dropdowns.forEach(dropdown => {
        const toggle = dropdown.querySelector('.dropdown-toggle');
        
        if (toggle) {
            toggle.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdown.classList.toggle('active');
                
                // Close other dropdowns
                dropdowns.forEach(other => {
                    if (other !== dropdown) {
                        other.classList.remove('active');
                    }
                });
            });
        }
    });
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', () => {
        dropdowns.forEach(dropdown => {
            dropdown.classList.remove('active');
        });
    });
});

// Format number with commas
MSP.formatNumber = (num) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
};
const formatNumber = MSP.formatNumber;

// Format percentage
MSP.formatPercentage = (num) => {
    return `${(num * 100).toFixed(1)}%`;
};
const formatPercentage = MSP.formatPercentage;

// Format time (ms to readable)
const formatTime = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
};
MSP.formatTime = formatTime;

// Get agent color
MSP.getAgentColor = (agentType) => {
    const colors = {
        'threat_intelligence': '#ef4444',
        'market_intelligence': '#3b82f6',
        'nlp_query': '#8b5cf6',
        'collaboration_matching': '#10b981',
        'client_health': '#f59e0b',
        'revenue_optimization': '#06b6d4',
        'anomaly_detection': '#ec4899',
        'security_compliance': '#14b8a6',
        'resource_allocation': '#6366f1',
        'federated_learning': '#a855f7'
    };
    return colors[agentType] || '#64748b';
};
const getAgentColor = MSP.getAgentColor;

// Get agent icon
MSP.getAgentIcon = (agentType) => {
    const icons = {
        'threat_intelligence': '🛡️',
        'market_intelligence': '💼',
        'nlp_query': '💬',
        'collaboration_matching': '🤝',
        'client_health': '📊',
        'revenue_optimization': '💰',
        'anomaly_detection': '🔍',
        'security_compliance': '✅',
        'resource_allocation': '📅',
        'federated_learning': '🌐'
    };
    return icons[agentType] || '🤖';
};
const getAgentIcon = MSP.getAgentIcon;

// Get agent name
MSP.getAgentName = (agentType) => {
    const names = {
        'threat_intelligence': 'Threat Intelligence',
        'market_intelligence': 'Market Intelligence',
        'nlp_query': 'NLP Query Assistant',
        'collaboration_matching': 'Collaboration Matching',
        'client_health': 'Client Health Prediction',
        'revenue_optimization': 'Revenue Optimization',
        'anomaly_detection': 'Anomaly Detection',
        'security_compliance': 'Security Compliance',
        'resource_allocation': 'Resource Allocation',
        'federated_learning': 'Federated Learning'
    };
    return names[agentType] || agentType;
};
const getAgentName = MSP.getAgentName;

// Create agent card
MSP.createAgentCard = (agentType, agentData) => {
    const icon = getAgentIcon(agentType);
    const name = getAgentName(agentType);
    const health = agentData.health_score * 100;
    const status = agentData.status;
    
    return `
        <div class="agent-card" onclick="navigateToAgent('${agentType}')">
            <div class="agent-card-icon">${icon}</div>
            <div class="agent-card-title">${name}</div>
            <div class="agent-card-status">
                <div class="status-indicator status-${status}"></div>
                <span>${status === 'active' ? 'Active' : 'Inactive'}</span>
            </div>
            <div class="agent-card-health">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.875rem; color: var(--gray);">Health</span>
                    <span style="font-weight: 600;">${health.toFixed(0)}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${health}%"></div>
                </div>
            </div>
        </div>
    `;
};

// Navigate to agent page
const navigateToAgent = (agentType) => {
    window.location.href = `${agentType.replace(/_/g, '-')}.html`;
};

// Example scenarios for each agent
const exampleScenarios = {
    threat_intelligence: [
        {
            name: 'Ransomware Attack',
            text: 'Detected suspicious file encryption activity across multiple endpoints. Possible ransomware attack in progress.'
        },
        {
            name: 'Phishing Attempt',
            text: 'Email with malicious attachment detected. Links to known phishing domain attempting credential theft.'
        },
        {
            name: 'DDoS Detection',
            text: 'Unusual spike in network traffic from multiple sources. Possible distributed denial of service attack.'
        }
    ],
    market_intelligence: [
        {
            name: 'Positive Market',
            text: 'Cloud adoption is accelerating. MSPs offering hybrid solutions are seeing 40% growth year-over-year.'
        },
        {
            name: 'Competitive Analysis',
            text: 'New competitor entered the market with aggressive pricing. Market share impact expected in Q2.'
        },
        {
            name: 'Industry Trends',
            text: 'AI integration becoming standard. MSPs without AI offerings losing clients to competitors.'
        }
    ],
    nlp_query: [
        {
            name: 'Revenue Query',
            text: 'What was our total revenue last quarter and how does it compare to the previous quarter?'
        },
        {
            name: 'Threat Summary',
            text: 'Summarize the top 5 threats detected this month and their severity levels.'
        },
        {
            name: 'Client Health',
            text: 'Which clients are at highest risk of churn and what can we do to retain them?'
        }
    ],
    collaboration_matching: [
        {
            name: 'Security Project',
            text: 'Need partner with SOC2 certification and 24/7 security operations for Fortune 500 client project.'
        },
        {
            name: 'Cloud Migration',
            text: 'Looking for MSP partner with Azure expertise and proven track record in healthcare industry.'
        },
        {
            name: 'Joint Proposal',
            text: 'Enterprise client needs comprehensive IT services. Seeking partners for infrastructure and support.'
        }
    ]
};

// Get example scenarios
const getExampleScenarios = (agentType) => {
    return exampleScenarios[agentType] || [];
};

// Display result in formatted card
MSP.displayResult = (containerId, data, title = 'Results') => {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    let html = `<div class="result-card animate-slide-in">`;
    html += `<h3 style="margin-bottom: 1rem;">${title}</h3>`;
    
    if (Array.isArray(data)) {
        data.forEach(item => {
            html += `<div class="result-item">
                <span class="result-label">${item.label}</span>
                <span class="result-value">${item.value}</span>
            </div>`;
        });
    } else if (typeof data === 'object') {
        Object.keys(data).forEach(key => {
            const value = typeof data[key] === 'object' ? 
                JSON.stringify(data[key], null, 2) : data[key];
            html += `<div class="result-item">
                <span class="result-label">${key.replace(/_/g, ' ').toUpperCase()}</span>
                <span class="result-value">${value}</span>
            </div>`;
        });
    }
    
    html += `</div>`;
    container.innerHTML = html;
};

// WebSocket connection for real-time updates
let ws = null;

const connectWebSocket = () => {
    try {
        ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onopen = () => {
            console.log('WebSocket connected');
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = () => {
            console.log('WebSocket closed. Reconnecting in 5s...');
            setTimeout(connectWebSocket, 5000);
        };
    } catch (error) {
        console.error('WebSocket connection failed:', error);
    }
};

const handleWebSocketMessage = (data) => {
    // Handle real-time updates
    if (data.type === 'agent_update') {
        updateAgentStatus(data.agent_type, data.status);
    } else if (data.type === 'activity') {
        addActivityItem(data);
    }
};

// Update agent status in real-time
const updateAgentStatus = (agentType, status) => {
    const agentCard = document.querySelector(`[data-agent="${agentType}"]`);
    if (agentCard) {
        const indicator = agentCard.querySelector('.status-indicator');
        if (indicator) {
            indicator.className = `status-indicator status-${status}`;
        }
    }
};

// Add activity item to feed
const addActivityItem = (activity) => {
    const feed = document.getElementById('activity-feed');
    if (!feed) return;
    
    const item = document.createElement('div');
    item.className = 'activity-item animate-slide-in';
    item.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span>${activity.icon || '🔔'}</span>
            <span>${activity.message}</span>
        </div>
        <div class="activity-time">${new Date().toLocaleTimeString()}</div>
    `;
    
    feed.insertBefore(item, feed.firstChild);
    
    // Keep only last 10 items
    while (feed.children.length > 10) {
        feed.removeChild(feed.lastChild);
    }
};

// Export functions
window.MSP = {
    apiCall,
    showLoading,
    hideLoading,
    showAlert,
    formatNumber,
    formatPercentage,
    formatTime,
    getAgentColor,
    getAgentIcon,
    getAgentName,
    createAgentCard,
    navigateToAgent,
    getExampleScenarios,
    displayResult,
    connectWebSocket
};

