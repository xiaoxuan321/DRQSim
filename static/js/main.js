/**
 * DRQSim - Dynamic Resource- and Queue-aware Process Simulation
 * Main JavaScript functionality
 */

// Global variables
let loadingModal;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap components
    initializeBootstrap();
    
    // Setup global event handlers
    setupGlobalHandlers();
    
    console.log('DRQSim initialized successfully');
});

function initializeBootstrap() {
    // Initialize loading modal
    const loadingModalElement = document.getElementById('loadingModal');
    if (loadingModalElement) {
        loadingModal = new bootstrap.Modal(loadingModalElement, {
            backdrop: 'static',
            keyboard: false
        });
    }
}

function setupGlobalHandlers() {
    // Handle file input changes
    document.addEventListener('change', function(e) {
        if (e.target.type === 'file') {
            handleFileInputChange(e.target);
        }
    });
    
    // Handle form submissions
    document.addEventListener('submit', function(e) {
        if (e.target.tagName === 'FORM') {
            // Prevent double submissions
            const submitBtn = e.target.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                setTimeout(() => {
                    submitBtn.disabled = false;
                }, 2000);
            }
        }
    });
}

function handleFileInputChange(input) {
    const file = input.files[0];
    if (!file) return;

    // Petri 网修改区的 PNML 上传框单独处理，不走左侧上传表单的校验逻辑
    if (input.id === 'modifiedPnmlFile') {
        if (!file.name.toLowerCase().endsWith('.pnml')) {
            showAlert('Please select a valid PNML file', 'warning');
            input.value = '';
            return;
        }

        if (file.size > 100 * 1024 * 1024) {
            showAlert('File size too large. Maximum size is 100MB', 'warning');
            input.value = '';
            return;
        }

        return;
    }

    // 只处理左侧主上传表单中的 XES / PNML
    if (input.id !== 'xesFile' && input.id !== 'pnmlFile') {
        return;
    }

    const discoverNet = document.getElementById('discoverNet');
    const importNet = document.getElementById('importNet');

    if (!discoverNet || !importNet) {
        return;
    }

    let valid = false;
    let requiredExtension = '';

    if (discoverNet.checked && file.name.toLowerCase().endsWith('.xes')) {
        valid = true;
        requiredExtension = 'XES';
    } else if (importNet.checked && file.name.toLowerCase().endsWith('.pnml')) {
        valid = true;
        requiredExtension = 'PNML';
    } else {
        if (discoverNet.checked) {
            requiredExtension = 'XES';
        } else if (importNet.checked) {
            requiredExtension = 'PNML';
        }
        showAlert(`Please select a valid ${requiredExtension} file`, 'warning');
        input.value = '';
        return;
    }
    
    // Check file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
        showAlert('File size too large. Maximum size is 100MB', 'warning');
        input.value = '';
        return;
    }
    
    // Update UI to show selected file
    const fileName = file.name;
    const fileInfo = document.createElement('small');
    fileInfo.className = 'text-muted d-block mt-1';
    fileInfo.textContent = `Selected: ${fileName} (${formatFileSize(file.size)})`;
    
    // Remove existing file info
    const existingInfo = input.parentNode.querySelector('small.text-muted');
    if (existingInfo && existingInfo.textContent.startsWith('Selected:')) {
        existingInfo.remove();
    }
    
    input.parentNode.appendChild(fileInfo);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

let discoveryLoadingStage = 'petri';
let discoveryProgressTimer = null;

const discoveryStageMessages = {
    petri: {
        title: 'Petri net mining...',
        detail: 'Discovering or validating the Petri net structure from the uploaded data.',
        label: 'Petri net mining'
    },
    params: {
        title: 'Parameter mining...',
        detail: 'Extracting resource, queue, arrival, execution-time, and waiting-time simulation parameters.',
        label: 'Parameter mining'
    }
};

function normalizeDiscoveryStage(stage) {
    if (!stage) return 'petri';

    const normalized = String(stage).toLowerCase();

    if ([
        'petri',
        'structure',
        'structure_search',
        'control_flow',
        'control-flow',
        'process_model',
        'pnml'
    ].includes(normalized)) {
        return 'petri';
    }

    if ([
        'params',
        'parameter',
        'parameters',
        'parameter_search',
        'runtime',
        'runtime_search',
        'simulation_parameters'
    ].includes(normalized)) {
        return 'params';
    }

    return 'petri';
}

function updateLoadingStage(stage = 'petri', shouldUpdateMessage = false) {
    discoveryLoadingStage = normalizeDiscoveryStage(stage);

    const stageBox = document.getElementById('loadingStageSingle');
    const stageText = document.getElementById('loadingStageText');
    const stageInfo = discoveryStageMessages[discoveryLoadingStage] || discoveryStageMessages.petri;

    if (stageBox) {
        stageBox.dataset.stage = discoveryLoadingStage;
    }

    if (stageText) {
        stageText.textContent = stageInfo.label;
    }

    if (shouldUpdateMessage) {
        updateLoadingMessage(stageInfo.title, stageInfo.detail);
    }
}

function startDiscoveryProgressPolling(sessionId, intervalMs = 1200) {
    stopDiscoveryProgressPolling();

    if (!sessionId) {
        console.warn('startDiscoveryProgressPolling called without sessionId.');
        updateLoadingStage('petri', true);
        return;
    }

    updateLoadingStage('petri', true);

    async function pollProgress() {
        try {
            const response = await fetch(`/api/discover_progress/${sessionId}?_=${Date.now()}`);
            const result = await response.json();

            if (!result.success) {
                return;
            }

            const stage = normalizeDiscoveryStage(result.stage || 'petri');
            const stageInfo = discoveryStageMessages[stage] || discoveryStageMessages.petri;

            updateLoadingStage(stage, false);
            updateLoadingMessage(
                result.title || stageInfo.title,
                result.detail || stageInfo.detail
            );

            if (result.status === 'ready' || result.status === 'error') {
                stopDiscoveryProgressPolling();
            }

        } catch (error) {
            console.warn('Failed to poll discovery progress:', error);
        }
    }

    pollProgress();
    discoveryProgressTimer = setInterval(pollProgress, intervalMs);
}

function stopDiscoveryProgressPolling() {
    if (discoveryProgressTimer) {
        clearInterval(discoveryProgressTimer);
        discoveryProgressTimer = null;
    }
}

/**
 * Backward-compatible wrapper.
 * Older templates may still call startDiscoveryLoadingStages(sessionId).
 * This version never alternates stages by timer; it only polls backend progress.
 */
function startDiscoveryLoadingStages(sessionId) {
    startDiscoveryProgressPolling(sessionId);
}

function stopDiscoveryLoadingStages() {
    stopDiscoveryProgressPolling();
}

function showLoading(title = 'Processing...', message = 'Please wait while we process your request.', options = {}) {
    if (!loadingModal) return;

    const titleElement = document.getElementById('loadingMessage');
    const messageElement = document.getElementById('loadingDetail');

    if (titleElement) titleElement.textContent = title;
    if (messageElement) messageElement.textContent = message;

    updateLoadingStage(options.stage || 'petri', false);
    loadingModal.show();
}

function hideLoading() {
    stopDiscoveryLoadingStages();
    if (loadingModal) {
        loadingModal.hide();
    }
}

function updateLoadingMessage(title, message) {
    const titleElement = document.getElementById('loadingMessage');
    const messageElement = document.getElementById('loadingDetail');

    if (titleElement) titleElement.textContent = title;
    if (messageElement) messageElement.textContent = message;
}

function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;

    // Success messages are important workflow feedback, so keep them visible longer.
    if (type === 'success' && duration === 5000) {
        duration = 7000;
    }

    const alertId = 'alert-' + Date.now();
    const alertHtml = `
        <div id="${alertId}" class="drq-center-alert drq-center-alert-${type}" role="alert" aria-live="polite">
            <div class="drq-center-alert-icon">
                <i data-feather="${getAlertIcon(type)}"></i>
            </div>
            <div class="drq-center-alert-body">
                <div class="drq-center-alert-title">${getAlertTitle(type)}</div>
                <div class="drq-center-alert-message">${message}</div>
            </div>
            <button type="button" class="btn-close drq-center-alert-close" aria-label="Close"></button>
        </div>
    `;

    alertContainer.innerHTML = alertHtml;

    const alertElement = document.getElementById(alertId);
    const closeBtn = alertElement.querySelector('.drq-center-alert-close');

    closeBtn.addEventListener('click', () => removeCenterAlert(alertElement));

    feather.replace();

    requestAnimationFrame(() => {
        alertElement.classList.add('show');
    });

    if (duration > 0) {
        setTimeout(() => {
            removeCenterAlert(alertElement);
        }, duration);
    }
}

function removeCenterAlert(alertElement) {
    if (!alertElement) return;
    alertElement.classList.remove('show');
    setTimeout(() => {
        if (alertElement.parentNode) {
            alertElement.remove();
        }
    }, 180);
}

function getAlertTitle(type) {
    const titles = {
        'success': 'Success',
        'danger': 'Error',
        'warning': 'Warning',
        'info': 'Information'
    };
    return titles[type] || 'Information';
}

function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'alert-circle',
        'warning': 'alert-triangle',
        'info': 'info'
    };
    return icons[type] || 'info';
}

function validateParameters(parameters) {
    const errors = [];

    // Validate structure
    const requiredSections = [
        'transition_params',
        'resource_params',
        'execution_time_params',
        'waiting_time_params'
    ];

    for (const section of requiredSections) {
        if (!parameters[section]) {
            errors.push(`Missing required section: ${section}`);
        }
    }

    // Validate transition parameters
    if (parameters.transition_params) {
        const weights = parameters.transition_params.transition_weights || {};
        for (const [transition, weight] of Object.entries(weights)) {
            if (typeof weight !== 'number' || weight < 0 || weight > 1) {
                errors.push(`Invalid transition weight for ${transition}: ${weight}`);
            }
        }
    }

    // Validate resource parameters
    if (parameters.resource_params) {
        const resourceWeights = parameters.resource_params.resource_weights || {};
        for (const [resource, weight] of Object.entries(resourceWeights)) {
            if (typeof weight !== 'number' || weight < 0 || weight > 1) {
                errors.push(`Invalid resource weight for ${resource}: ${weight}`);
            }
        }
    }

    // Validate execution time parameters
    if (parameters.execution_time_params) {
        const durations = parameters.execution_time_params.activity_durations || {};
        for (const [activity, params] of Object.entries(durations)) {
            if (!params.distribution || !params.mean) {
                errors.push(`Invalid duration parameters for ${activity}`);
            }
            if (params.mean <= 0) {
                errors.push(`Mean duration must be positive for ${activity}`);
            }
            if (params.min && params.max && params.min >= params.max) {
                errors.push(`Minimum duration must be less than maximum for ${activity}`);
            }
        }
    }

    return errors;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Utility functions for parameter manipulation
function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function formatDateTime(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(new Date(date));
}

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Export functions for global use
window.DRQSim = {
    showLoading,
    hideLoading,
    updateLoadingMessage,
    updateLoadingStage,
    startDiscoveryProgressPolling,
    stopDiscoveryProgressPolling,
    startDiscoveryLoadingStages,
    stopDiscoveryLoadingStages,
    showAlert,
    validateParameters,
    debounce,
    throttle,
    deepClone,
    downloadJSON,
    formatDateTime,
    formatFileSize,
    generateId
};


// Backward-compatible alias for older templates/scripts.
window.ProSiT = window.DRQSim;
