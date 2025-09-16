// RFAI Dashboard JavaScript

// System data
const systemData = {
    "system_status": "OPERATIONAL",
    "performance": {
        "current": 0.325,
        "baseline": 0.153,
        "improvement": 112.3,
        "tasks_completed": 5,
        "success_rate": 100
    },
    "fractal_hierarchy": {
        "levels": 4,
        "total_params": 37440,
        "level_params": [32768, 4096, 512, 64],
        "level_modules": [8, 4, 2, 1]
    },
    "agents": [
        {"id": "agent_001", "name": "Pattern Recognition Alpha", "specialization": "pattern_recognition", "completion_rate": 0.431, "efficiency": 0.388, "status": "active"},
        {"id": "agent_002", "name": "Pattern Recognition Beta", "specialization": "pattern_recognition", "completion_rate": 0.431, "efficiency": 0.388, "status": "active"},
        {"id": "agent_003", "name": "Optimization Prime", "specialization": "optimization", "completion_rate": 0.342, "efficiency": 0.470, "status": "active"},
        {"id": "agent_004", "name": "Optimization Sigma", "specialization": "optimization", "completion_rate": 0.342, "efficiency": 0.470, "status": "standby"},
        {"id": "agent_005", "name": "Memory Manager Alpha", "specialization": "memory_management", "completion_rate": 0.596, "efficiency": 0.544, "status": "active"},
        {"id": "agent_006", "name": "Memory Manager Beta", "specialization": "memory_management", "completion_rate": 0.596, "efficiency": 0.544, "status": "active"},
        {"id": "agent_007", "name": "Goal Planner Prime", "specialization": "goal_planning", "completion_rate": 0.434, "efficiency": 0.495, "status": "active"},
        {"id": "agent_008", "name": "Goal Planner Sigma", "specialization": "goal_planning", "completion_rate": 0.434, "efficiency": 0.495, "status": "standby"},
        {"id": "agent_009", "name": "Resource Allocator", "specialization": "resource_allocation", "completion_rate": 0.534, "efficiency": 0.384, "status": "active"},
        {"id": "agent_010", "name": "Conflict Resolver", "specialization": "conflict_resolution", "completion_rate": 0.486, "efficiency": 0.462, "status": "standby"},
        {"id": "agent_011", "name": "Learning Coordinator", "specialization": "learning_coordination", "completion_rate": 0.518, "efficiency": 0.410, "status": "active"},
        {"id": "agent_012", "name": "Emergent Detector", "specialization": "emergent_behavior_detection", "completion_rate": 0.317, "efficiency": 0.432, "status": "active"}
    ],
    "performance_history": [0.153, 0.181, 0.199, 0.217, 0.226, 0.243, 0.263, 0.280, 0.289, 0.303, 0.325],
    "quantum_status": {
        "enabled": true,
        "qubits": 16,
        "coherence": 0.847,
        "error_rate": 0.023
    },
    "logs": [
        "System initialized successfully",
        "Fractal hierarchy built with 4 levels",
        "12 autonomous agents deployed",
        "Quantum processor connected (16 qubits)",
        "Task fractal_pattern_001 completed - Performance: 0.166",
        "Task recursive_opt_002 completed - Performance: 0.135",
        "Meta-learning optimization triggered",
        "Architecture evolution completed",
        "Performance improvement: +112.3%",
        "Emergent behavior detected: controlled evolution",
        "System status: OPTIMAL"
    ]
};

// Global variables
let performanceChart;
let logEntries = [...systemData.logs];
let isAutoScroll = true;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startRealTimeUpdates();
});

function initializeDashboard() {
    populateHierarchyLevels();
    createSwarmVisualization();
    initializePerformanceChart();
    populateAgentGrid();
    createParameterVisualization();
    populateSystemLogs();
    updateActiveAgentCount();
}

function populateHierarchyLevels() {
    const hierarchyContainer = document.getElementById('hierarchyLevels');
    const levels = ['Global', 'Regional', 'Local', 'Micro'];
    
    hierarchyContainer.innerHTML = '';
    
    levels.forEach((level, index) => {
        const levelDiv = document.createElement('div');
        levelDiv.className = 'hierarchy-level';
        levelDiv.innerHTML = `
            <div class="level-info">
                <div class="level-name">Level ${index + 1}: ${level}</div>
                <div class="level-params">${systemData.fractal_hierarchy.level_params[index].toLocaleString()} params</div>
            </div>
            <div class="level-modules">${systemData.fractal_hierarchy.level_modules[index]} modules</div>
        `;
        hierarchyContainer.appendChild(levelDiv);
    });
}

function createSwarmVisualization() {
    const swarmContainer = document.getElementById('swarmViz');
    swarmContainer.innerHTML = '';
    
    systemData.agents.forEach((agent, index) => {
        const agentNode = document.createElement('div');
        agentNode.className = `agent-node ${agent.status}`;
        agentNode.title = `${agent.name} - ${agent.specialization.replace('_', ' ')}`;
        agentNode.addEventListener('click', () => highlightAgent(agent.id));
        swarmContainer.appendChild(agentNode);
    });
}

function initializePerformanceChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: systemData.performance_history.map((_, index) => `Iter ${index + 1}`),
            datasets: [{
                label: 'Performance Score',
                data: systemData.performance_history,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#00d4ff',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.1,
                    max: 0.4,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a8b8'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a8b8'
                    }
                }
            },
            elements: {
                point: {
                    hoverBackgroundColor: '#00ff88'
                }
            }
        }
    });
}

function populateAgentGrid() {
    const agentGrid = document.getElementById('agentGrid');
    agentGrid.innerHTML = '';
    
    systemData.agents.forEach(agent => {
        const agentCard = document.createElement('div');
        agentCard.className = `agent-card ${agent.status}`;
        agentCard.id = `card-${agent.id}`;
        
        agentCard.innerHTML = `
            <div class="agent-name">${agent.name}</div>
            <div class="agent-specialization">${agent.specialization.replace(/_/g, ' ')}</div>
            <div class="agent-metrics">
                <div class="agent-metric">
                    <span>Completion:</span>
                    <span>${(agent.completion_rate * 100).toFixed(1)}%</span>
                </div>
                <div class="agent-metric">
                    <span>Efficiency:</span>
                    <span>${(agent.efficiency * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;
        
        agentCard.addEventListener('click', () => highlightAgent(agent.id));
        agentGrid.appendChild(agentCard);
    });
}

function createParameterVisualization() {
    const paramLevels = document.getElementById('paramLevels');
    const levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4'];
    const maxParams = Math.max(...systemData.fractal_hierarchy.level_params);
    
    paramLevels.innerHTML = '';
    
    levels.forEach((level, index) => {
        const params = systemData.fractal_hierarchy.level_params[index];
        const percentage = (params / maxParams) * 100;
        
        const paramLevel = document.createElement('div');
        paramLevel.className = 'param-level';
        paramLevel.innerHTML = `
            <div class="param-label">${level}</div>
            <div class="param-bar">
                <div class="param-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="param-value">${params.toLocaleString()}</div>
        `;
        
        paramLevels.appendChild(paramLevel);
    });
}

function populateSystemLogs() {
    const logOutput = document.getElementById('logOutput');
    logOutput.innerHTML = '';
    
    logEntries.forEach((log, index) => {
        addLogEntry(log, getLogType(log));
    });
    
    if (isAutoScroll) {
        logOutput.scrollTop = logOutput.scrollHeight;
    }
}

function addLogEntry(message, type = 'info') {
    const logOutput = document.getElementById('logOutput');
    const timestamp = new Date().toLocaleTimeString();
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-message">${message}</span>
    `;
    
    logOutput.appendChild(logEntry);
    
    if (isAutoScroll) {
        logOutput.scrollTop = logOutput.scrollHeight;
    }
}

function getLogType(message) {
    if (message.includes('completed') || message.includes('success') || message.includes('OPTIMAL')) {
        return 'success';
    } else if (message.includes('warning') || message.includes('detected')) {
        return 'warning';
    } else if (message.includes('error') || message.includes('failed')) {
        return 'error';
    }
    return 'info';
}

function highlightAgent(agentId) {
    // Remove previous highlights
    document.querySelectorAll('.agent-card').forEach(card => {
        card.style.transform = '';
        card.style.boxShadow = '';
    });
    
    // Highlight selected agent
    const agentCard = document.getElementById(`card-${agentId}`);
    if (agentCard) {
        agentCard.style.transform = 'scale(1.05)';
        agentCard.style.boxShadow = '0 0 20px rgba(0, 212, 255, 0.5)';
        
        // Find agent data and add log entry
        const agent = systemData.agents.find(a => a.id === agentId);
        if (agent) {
            addLogEntry(`Agent selected: ${agent.name} (${agent.specialization.replace(/_/g, ' ')})`, 'info');
            logEntries.push(`Agent selected: ${agent.name}`);
        }
    }
}

function updateActiveAgentCount() {
    const activeCount = systemData.agents.filter(agent => agent.status === 'active').length;
    document.getElementById('activeAgents').textContent = `${activeCount}/12`;
}

function setupEventListeners() {
    // Control buttons
    document.getElementById('simulateTask').addEventListener('click', simulateTask);
    document.getElementById('triggerOptimization').addEventListener('click', triggerOptimization);
    document.getElementById('adjustParams').addEventListener('click', adjustParameters);
    document.getElementById('emergencyStop').addEventListener('click', emergencyStop);
    
    // Log controls
    document.getElementById('clearLogs').addEventListener('click', clearLogs);
    document.getElementById('exportLogs').addEventListener('click', exportLogs);
    document.getElementById('autoScroll').addEventListener('change', function() {
        isAutoScroll = this.checked;
    });
}

function simulateTask() {
    const taskId = `task_${Date.now()}`;
    const performance = (Math.random() * 0.1 + 0.25).toFixed(3);
    
    addLogEntry(`Simulating new task: ${taskId}`, 'info');
    
    // Animate some agents
    animateAgentActivity();
    
    setTimeout(() => {
        addLogEntry(`Task ${taskId} completed - Performance: ${performance}`, 'success');
        logEntries.push(`Task ${taskId} completed`);
        
        // Update performance metrics
        const newTaskCount = parseInt(document.getElementById('tasksCompleted').textContent) + 1;
        document.getElementById('tasksCompleted').textContent = newTaskCount;
        
        // Update performance chart
        systemData.performance_history.push(parseFloat(performance));
        if (systemData.performance_history.length > 15) {
            systemData.performance_history.shift();
        }
        updatePerformanceChart();
    }, 2000);
}

function triggerOptimization() {
    addLogEntry('Meta-learning optimization initiated...', 'warning');
    
    const optimizationSteps = [
        'Analyzing current performance patterns',
        'Identifying optimization opportunities',
        'Applying fractal parameter adjustments',
        'Validating optimization results'
    ];
    
    optimizationSteps.forEach((step, index) => {
        setTimeout(() => {
            addLogEntry(step, 'info');
        }, (index + 1) * 1000);
    });
    
    setTimeout(() => {
        addLogEntry('Optimization completed - Performance improved by 3.2%', 'success');
        logEntries.push('Optimization completed');
        animateParameterUpdate();
    }, 5000);
}

function adjustParameters() {
    addLogEntry('Parameter adjustment interface activated', 'info');
    addLogEntry('Manual parameter tuning enabled - Use with caution', 'warning');
    
    // Animate parameter bars
    const paramFills = document.querySelectorAll('.param-fill');
    paramFills.forEach(fill => {
        fill.style.transition = 'width 1s ease-in-out';
        const currentWidth = parseInt(fill.style.width);
        const newWidth = Math.max(10, Math.min(100, currentWidth + (Math.random() - 0.5) * 20));
        fill.style.width = `${newWidth}%`;
    });
}

function emergencyStop() {
    addLogEntry('EMERGENCY STOP ACTIVATED', 'error');
    addLogEntry('All agent operations halted', 'error');
    addLogEntry('System entering safe mode...', 'warning');
    
    // Update status indicator
    const statusIndicator = document.getElementById('statusIndicator');
    statusIndicator.textContent = 'EMERGENCY STOP';
    statusIndicator.className = 'status-indicator error';
    statusIndicator.style.background = 'rgba(255, 68, 68, 0.2)';
    statusIndicator.style.color = '#ff4444';
    statusIndicator.style.borderColor = '#ff4444';
    
    setTimeout(() => {
        addLogEntry('System recovering from emergency stop...', 'info');
        addLogEntry('Resuming normal operations', 'success');
        statusIndicator.textContent = 'OPERATIONAL';
        statusIndicator.className = 'status-indicator operational';
        statusIndicator.style.background = 'rgba(0, 255, 136, 0.2)';
        statusIndicator.style.color = '#00ff88';
        statusIndicator.style.borderColor = '#00ff88';
    }, 3000);
}

function clearLogs() {
    document.getElementById('logOutput').innerHTML = '';
    logEntries = [];
    addLogEntry('System logs cleared', 'info');
}

function exportLogs() {
    const logData = logEntries.join('\n');
    const blob = new Blob([logData], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rfai_logs_${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    window.URL.revokeObjectURL(url);
    
    addLogEntry('Logs exported successfully', 'success');
}

function animateAgentActivity() {
    const agentNodes = document.querySelectorAll('.agent-node.active');
    agentNodes.forEach(node => {
        node.style.animation = 'none';
        node.offsetHeight; // Trigger reflow
        node.style.animation = 'agentPulse 0.5s ease-in-out 3';
    });
}

function animateParameterUpdate() {
    const paramFills = document.querySelectorAll('.param-fill');
    paramFills.forEach((fill, index) => {
        setTimeout(() => {
            fill.style.transition = 'width 0.8s ease-out';
            const currentWidth = parseInt(fill.style.width);
            const improvement = Math.random() * 10 + 5;
            fill.style.width = `${Math.min(100, currentWidth + improvement)}%`;
        }, index * 200);
    });
}

function updatePerformanceChart() {
    if (performanceChart) {
        performanceChart.data.labels = systemData.performance_history.map((_, index) => `Iter ${index + 1}`);
        performanceChart.data.datasets[0].data = systemData.performance_history;
        performanceChart.update();
        
        // Update current performance display
        const current = systemData.performance_history[systemData.performance_history.length - 1];
        document.getElementById('currentPerformance').textContent = current.toFixed(3);
    }
}

function startRealTimeUpdates() {
    // Simulate real-time system activity
    setInterval(() => {
        // Randomly add system activity logs
        if (Math.random() > 0.7) {
            const activities = [
                'Fractal optimization cycle completed',
                'Agent swarm rebalancing initiated',
                'Quantum coherence maintained at optimal levels',
                'Emergent pattern detected in data stream',
                'Resource allocation optimized',
                'Neural pathway strengthening observed'
            ];
            
            const activity = activities[Math.floor(Math.random() * activities.length)];
            addLogEntry(activity, Math.random() > 0.8 ? 'success' : 'info');
            logEntries.push(activity);
        }
        
        // Update quantum metrics with small variations
        if (Math.random() > 0.8) {
            updateQuantumMetrics();
        }
    }, 5000);
    
    // Update agent activity animation
    setInterval(() => {
        animateRandomAgents();
    }, 3000);
}

function updateQuantumMetrics() {
    const coherence = (0.847 + (Math.random() - 0.5) * 0.02).toFixed(3);
    const errorRate = (0.023 + (Math.random() - 0.5) * 0.005).toFixed(3);
    
    document.querySelector('.quantum-metrics .quantum-metric:nth-child(2) .quantum-value').textContent = `${(coherence * 100).toFixed(1)}%`;
    document.querySelector('.quantum-metrics .quantum-metric:nth-child(3) .quantum-value').textContent = `${(errorRate * 100).toFixed(1)}%`;
}

function animateRandomAgents() {
    const activeAgents = document.querySelectorAll('.agent-node.active');
    const randomCount = Math.floor(Math.random() * 3) + 1;
    
    for (let i = 0; i < randomCount; i++) {
        const randomAgent = activeAgents[Math.floor(Math.random() * activeAgents.length)];
        if (randomAgent) {
            randomAgent.style.transform = 'scale(1.3)';
            randomAgent.style.boxShadow = '0 0 15px rgba(0, 255, 136, 0.8)';
            
            setTimeout(() => {
                randomAgent.style.transform = '';
                randomAgent.style.boxShadow = '';
            }, 800);
        }
    }
}