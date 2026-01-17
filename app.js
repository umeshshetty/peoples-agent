/**
 * People's Agent - AI Second Brain
 * Main application logic with Brain World dashboard
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements - Chat View
const thoughtInput = document.getElementById('thought-input');
const charCount = document.getElementById('char-count');
const submitBtn = document.getElementById('submit-btn');
const thoughtsList = document.getElementById('thoughts-list');
const themeToggle = document.getElementById('theme-toggle');

// DOM Elements - Views
const chatView = document.getElementById('chat-view');
const brainWorld = document.getElementById('brain-world');
const viewTabs = document.querySelectorAll('.view-tab');

// DOM Elements - Brain World
const brainStats = document.getElementById('brain-stats');
const categoriesGrid = document.getElementById('categories-grid');
const categoryDetail = document.getElementById('category-detail');
const detailTitle = document.getElementById('detail-title');
const detailItems = document.getElementById('detail-items');
const backBtn = document.getElementById('back-to-categories');

// State
let thoughts = [];
let isProcessing = false;
let currentView = 'chat';
let brainInsights = null;

// Initialize the application
function init() {
    loadThoughts();
    loadTheme();
    setupEventListeners();
    renderThoughts();
    focusInput();
}

// Event Listeners
function setupEventListeners() {
    // Character count update
    thoughtInput.addEventListener('input', handleInputChange);

    // Submit thought
    submitBtn.addEventListener('click', handleSubmit);

    // Submit with keyboard shortcut (Cmd/Ctrl + Enter)
    thoughtInput.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            handleSubmit();
        }
    });

    // Theme toggle
    themeToggle.addEventListener('click', toggleTheme);

    // Auto-resize textarea
    thoughtInput.addEventListener('input', autoResize);

    // View switching
    viewTabs.forEach(tab => {
        tab.addEventListener('click', () => switchView(tab.dataset.view));
    });

    // Back button in category detail
    if (backBtn) {
        backBtn.addEventListener('click', showCategoriesGrid);
    }
}

// Handle input changes
function handleInputChange() {
    const length = thoughtInput.value.length;
    const max = thoughtInput.getAttribute('maxlength');
    charCount.textContent = `${length.toLocaleString()} / ${parseInt(max).toLocaleString()}`;

    // Visual feedback when approaching limit
    if (length > max * 0.9) {
        charCount.style.color = '#ef4444';
    } else if (length > max * 0.75) {
        charCount.style.color = '#f59e0b';
    } else {
        charCount.style.color = '';
    }
}

// Auto-resize textarea
function autoResize() {
    thoughtInput.style.height = 'auto';
    thoughtInput.style.height = Math.min(thoughtInput.scrollHeight, 400) + 'px';
}

// Handle thought submission
async function handleSubmit() {
    const content = thoughtInput.value.trim();

    if (!content) {
        shakeButton();
        return;
    }

    if (isProcessing) return;

    isProcessing = true;
    setButtonLoading(true);

    // Create new thought object
    const thought = {
        id: generateId(),
        content: content,
        timestamp: new Date().toISOString(),
        aiResponse: null,
        aiAnalysis: null,
        aiInsights: null,
        entities: [],
        categories: [],
        summary: '',
        hasConnections: false,
    };

    // Add to thoughts array and render immediately
    thoughts.push(thought);  // Add at the end (bottom) not beginning
    saveThoughts();
    renderThoughts();

    // Clear input
    thoughtInput.value = '';
    thoughtInput.style.height = '';
    handleInputChange();

    // Show processing state in the thought card
    showThoughtProcessing(thought.id);

    try {
        // Call the AI backend
        const response = await fetch(`${API_BASE_URL}/api/think`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ thought: content }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        // Update the thought with AI response and knowledge graph data
        thought.aiResponse = result.response;
        thought.aiAnalysis = result.analysis;
        thought.aiInsights = result.insights;
        thought.entities = result.entities || [];
        thought.categories = result.categories || [];
        thought.summary = result.summary || '';
        thought.hasConnections = result.has_connections || false;

        // Update in array and re-render
        const idx = thoughts.findIndex(t => t.id === thought.id);
        if (idx !== -1) {
            thoughts[idx] = thought;
        }
        saveThoughts();
        renderThoughts();

        showSubmitSuccess();

    } catch (error) {
        console.error('Error calling AI:', error);
        showThoughtError(thought.id, error.message);
    } finally {
        isProcessing = false;
        setButtonLoading(false);
        thoughtInput.focus();
    }
}

// Show processing state in thought card
function showThoughtProcessing(id) {
    const card = document.querySelector(`[data-id="${id}"]`);
    if (card) {
        const responseArea = card.querySelector('.thought-response');
        if (responseArea) {
            responseArea.innerHTML = `
                <div class="ai-thinking">
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span>AI is thinking...</span>
                </div>
            `;
        }
    }
}

// Show error in thought card
function showThoughtError(id, message) {
    const card = document.querySelector(`[data-id="${id}"]`);
    if (card) {
        const responseArea = card.querySelector('.thought-response');
        if (responseArea) {
            responseArea.innerHTML = `
                <div class="ai-error">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <span>Error: ${message}. Make sure backend is running at http://localhost:8000</span>
                    <button class="retry-btn" onclick="retryThought('${id}')">Retry</button>
                </div>
            `;
        }
    }
}

// Retry a failed thought
async function retryThought(id) {
    const thought = thoughts.find(t => t.id === id);
    if (!thought) return;

    showThoughtProcessing(id);

    try {
        const response = await fetch(`${API_BASE_URL}/api/think`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ thought: thought.content }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        thought.aiResponse = result.response;
        thought.aiAnalysis = result.analysis;
        thought.aiInsights = result.insights;

        saveThoughts();
        renderThoughts();

    } catch (error) {
        showThoughtError(id, error.message);
    }
}

// Set button loading state
function setButtonLoading(loading) {
    if (loading) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <span class="btn-text">Processing...</span>
            <div class="btn-spinner"></div>
        `;
    } else {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <span class="btn-text">Send to AI</span>
            <svg class="btn-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 2L11 13"/>
                <path d="M22 2L15 22L11 13L2 9L22 2Z"/>
            </svg>
            <div class="btn-shine"></div>
        `;
    }
}

// Generate unique ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Shake button animation for empty submit
function shakeButton() {
    submitBtn.style.animation = 'shake 0.5s ease';
    submitBtn.addEventListener('animationend', () => {
        submitBtn.style.animation = '';
    }, { once: true });

    // Add shake animation to stylesheet if not exists
    if (!document.querySelector('#shake-animation')) {
        const style = document.createElement('style');
        style.id = 'shake-animation';
        style.textContent = `
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                20% { transform: translateX(-8px); }
                40% { transform: translateX(8px); }
                60% { transform: translateX(-4px); }
                80% { transform: translateX(4px); }
            }
        `;
        document.head.appendChild(style);
    }
}

// Show submit success animation
function showSubmitSuccess() {
    submitBtn.innerHTML = `
        <span class="btn-text">Sent!</span>
        <svg class="btn-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20,6 9,17 4,12"/>
        </svg>
        <div class="btn-shine"></div>
    `;

    setTimeout(() => {
        setButtonLoading(false);
    }, 1500);
}

// Render thoughts list
function renderThoughts() {
    if (thoughts.length === 0) {
        thoughtsList.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                    </svg>
                </div>
                <p>Your thoughts will appear here</p>
                <span class="empty-hint">Start typing above to capture your first thought</span>
            </div>
        `;
        return;
    }

    thoughtsList.innerHTML = thoughts.map(thought => `
        <div class="thought-card" data-id="${thought.id}">
            <div class="thought-card-header">
                <span class="thought-time">${formatTime(thought.timestamp)}</span>
                <button class="thought-delete" onclick="deleteThought('${thought.id}')" title="Delete thought">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                    </svg>
                </button>
            </div>
            <div class="thought-content user-thought">
                <div class="thought-avatar user-avatar">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                        <circle cx="12" cy="7" r="4"/>
                    </svg>
                </div>
                <div class="thought-bubble">${escapeHtml(thought.content)}</div>
            </div>
            ${renderKnowledgeTags(thought)}
            <div class="thought-response">
                ${thought.aiResponse ? `
                    <div class="thought-content ai-thought">
                        <div class="thought-avatar ai-avatar">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M12 8v4M12 16h.01"/>
                            </svg>
                        </div>
                        <div class="thought-bubble ai-bubble">
                            ${thought.hasConnections ? '<span class="connection-badge" title="Connected to previous thoughts">🔗</span>' : ''}
                            ${escapeHtml(thought.aiResponse)}
                        </div>
                    </div>
                ` : `
                    <div class="ai-thinking">
                        <div class="thinking-dots">
                            <span></span><span></span><span></span>
                        </div>
                        <span>Analyzing and storing in knowledge graph...</span>
                    </div>
                `}
            </div>
        </div>
    `).join('');
}

// Format timestamp
function formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
    });
}

// Render knowledge tags (entities and categories)
function renderKnowledgeTags(thought) {
    const entities = thought.entities || [];
    const categories = thought.categories || [];

    if (entities.length === 0 && categories.length === 0) {
        return '';
    }

    const entityIcons = {
        'Person': '👤',
        'Place': '📍',
        'Topic': '💡',
        'Project': '📁',
        'Concept': '🧠',
        'Organization': '🏢',
        'Tool': '🔧',
        'Skill': '⭐'
    };

    const categoryColors = {
        'Work': 'var(--accent-purple)',
        'Personal': 'var(--accent-pink)',
        'Ideas': 'var(--accent-cyan)',
        'Goals': '#22c55e',
        'Tasks': '#f59e0b',
        'Questions': '#6366f1',
        'Learning': '#14b8a6',
        'Reflection': '#a855f7'
    };

    let html = '<div class="knowledge-tags">';

    // Render categories
    if (categories.length > 0) {
        html += '<div class="tag-group">';
        categories.forEach(cat => {
            const color = categoryColors[cat.name] || 'var(--text-muted)';
            html += `<span class="category-tag" style="--tag-color: ${color}">${cat.name}</span>`;
        });
        html += '</div>';
    }

    // Render entities
    if (entities.length > 0) {
        html += '<div class="tag-group">';
        entities.forEach(entity => {
            const icon = entityIcons[entity.type] || '📌';
            html += `<span class="entity-tag" title="${entity.type}: ${entity.description || entity.name}">${icon} ${entity.name}</span>`;
        });
        html += '</div>';
    }

    html += '</div>';
    return html;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Delete a thought
function deleteThought(id) {
    const card = document.querySelector(`[data-id="${id}"]`);
    if (card) {
        card.style.animation = 'slide-out 0.3s ease forwards';
        setTimeout(() => {
            thoughts = thoughts.filter(t => t.id !== id);
            saveThoughts();
            renderThoughts();
        }, 300);
    }

    // Add slide-out animation if not exists
    if (!document.querySelector('#slide-out-animation')) {
        const style = document.createElement('style');
        style.id = 'slide-out-animation';
        style.textContent = `
            @keyframes slide-out {
                to {
                    opacity: 0;
                    transform: translateX(-20px);
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Local Storage operations
function saveThoughts() {
    localStorage.setItem('peoples-agent-thoughts', JSON.stringify(thoughts));
}

function loadThoughts() {
    try {
        const saved = localStorage.getItem('peoples-agent-thoughts');
        thoughts = saved ? JSON.parse(saved) : [];
    } catch (e) {
        console.error('Error loading thoughts:', e);
        thoughts = [];
    }
}

// Theme management
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('peoples-agent-theme', newTheme);

    // Add rotation animation to button
    themeToggle.style.transform = 'rotate(360deg)';
    setTimeout(() => {
        themeToggle.style.transform = '';
    }, 300);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('peoples-agent-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');

    document.documentElement.setAttribute('data-theme', theme);
}

// Focus input on load
function focusInput() {
    setTimeout(() => {
        thoughtInput.focus();
    }, 100);
}

// ============================================================================
// Brain World Functions
// ============================================================================

// Switch between Chat and Brain World views
function switchView(view) {
    currentView = view;

    // Update tabs
    viewTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });

    // Update views
    if (view === 'chat') {
        chatView.classList.add('active');
        brainWorld.classList.remove('active');
    } else {
        chatView.classList.remove('active');
        brainWorld.classList.add('active');
        loadBrainWorld();
    }
}

// Load Brain World data
async function loadBrainWorld() {
    try {
        // Load stats and insights
        const [statsRes, insightsRes] = await Promise.all([
            fetch(`${API_BASE_URL}/api/brain/stats`),
            fetch(`${API_BASE_URL}/api/brain/insights`)
        ]);

        const statsData = await statsRes.json();
        const insightsData = await insightsRes.json();

        brainInsights = insightsData.insights;

        renderBrainStats(statsData.stats);
        renderCategoriesGrid(brainInsights);
        showCategoriesGrid();

    } catch (error) {
        console.error('Error loading brain world:', error);
        categoriesGrid.innerHTML = `
            <div class="brain-empty">
                <div class="brain-empty-icon">🧠</div>
                <h3>Couldn't load Brain World</h3>
                <p>Make sure the backend is running</p>
            </div>
        `;
    }
}

// Render brain stats
function renderBrainStats(stats) {
    brainStats.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${stats.total_thoughts || 0}</div>
            <div class="stat-label">Notes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.total_entities || 0}</div>
            <div class="stat-label">Entities</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.total_conversations || 0}</div>
            <div class="stat-label">Messages</div>
        </div>
    `;
}

// Render categories grid
function renderCategoriesGrid(insights) {
    if (!insights || Object.keys(insights).length === 0) {
        categoriesGrid.innerHTML = `
            <div class="brain-empty">
                <div class="brain-empty-icon">🧠</div>
                <h3>Your brain is empty!</h3>
                <p>Start adding notes in the Chat view to populate your brain</p>
            </div>
        `;
        return;
    }

    const categoryOrder = ['urgent', 'people', 'projects', 'meetings', 'tasks', 'ideas', 'learning', 'reflections'];

    categoriesGrid.innerHTML = categoryOrder.map(key => {
        const cat = insights[key];
        if (!cat) return '';

        const previewItems = cat.items.slice(0, 3);
        const previewHtml = previewItems.length > 0
            ? previewItems.map(item => `<div class="preview-item">${escapeHtml(item.summary || item.content)}</div>`).join('')
            : '<div class="category-empty">No items yet</div>';

        return `
            <div class="category-card" data-category="${key}" onclick="showCategoryDetail('${key}')">
                <div class="category-header">
                    <span class="category-icon">${cat.icon}</span>
                    <span class="category-label">${cat.label}</span>
                    <span class="category-count">${cat.count}</span>
                </div>
                <div class="category-preview">
                    ${previewHtml}
                </div>
            </div>
        `;
    }).join('');
}

// Show category detail view
async function showCategoryDetail(categoryKey) {
    if (!brainInsights || !brainInsights[categoryKey]) return;

    const cat = brainInsights[categoryKey];
    detailTitle.innerHTML = `<span>${cat.icon}</span> ${cat.label}`;

    // Special handling for People, Projects, Meetings - fetch synthesized data
    if (categoryKey === 'people') {
        try {
            const res = await fetch(`${API_BASE_URL}/api/brain/people`);
            const data = await res.json();
            renderPeopleProfiles(data.people || []);
        } catch (e) {
            renderGenericItems(cat.items);
        }
    } else if (categoryKey === 'projects') {
        try {
            const res = await fetch(`${API_BASE_URL}/api/brain/projects`);
            const data = await res.json();
            renderProjectProfiles(data.projects || []);
        } catch (e) {
            renderGenericItems(cat.items);
        }
    } else if (categoryKey === 'meetings') {
        try {
            const res = await fetch(`${API_BASE_URL}/api/brain/meetings`);
            const data = await res.json();
            renderMeetings(data.meetings || []);
        } catch (e) {
            renderGenericItems(cat.items);
        }
    } else {
        renderGenericItems(cat.items);
    }

    categoriesGrid.style.display = 'none';
    brainStats.style.display = 'none';
    categoryDetail.classList.add('active');
}

// Render synthesized person profiles
function renderPeopleProfiles(people) {
    if (people.length === 0) {
        detailItems.innerHTML = `<div class="brain-empty"><p>No people profiles yet</p></div>`;
        return;
    }
    detailItems.innerHTML = people.map(p => `
        <div class="detail-item profile-card">
            <div class="profile-header">
                <span class="profile-avatar">👤</span>
                <div class="profile-info">
                    <div class="profile-name">${escapeHtml(p.name)}</div>
                    <div class="profile-role">${escapeHtml(p.role)} • ${escapeHtml(p.relationship)}</div>
                </div>
                <span class="profile-mentions">${p.mention_count || 0} mentions</span>
            </div>
            <div class="profile-summary">${escapeHtml(p.summary)}</div>
            ${p.topics && p.topics.length ? `<div class="profile-topics">${p.topics.map(t => `<span class="topic-tag">${t}</span>`).join('')}</div>` : ''}
        </div>
    `).join('');
}

// Render synthesized project profiles
function renderProjectProfiles(projects) {
    if (projects.length === 0) {
        detailItems.innerHTML = `<div class="brain-empty"><p>No project profiles yet</p></div>`;
        return;
    }
    detailItems.innerHTML = projects.map(p => `
        <div class="detail-item profile-card">
            <div class="profile-header">
                <span class="profile-avatar">📁</span>
                <div class="profile-info">
                    <div class="profile-name">${escapeHtml(p.name)}</div>
                    <div class="profile-role">Status: ${escapeHtml(p.status)}</div>
                </div>
                <span class="profile-mentions">${p.mention_count || 0} mentions</span>
            </div>
            <div class="profile-summary">${escapeHtml(p.summary)}</div>
            ${p.deadline ? `<div class="profile-deadline">📅 Deadline: ${escapeHtml(p.deadline)}</div>` : ''}
            ${p.people && p.people.length ? `<div class="profile-team">👥 Team: ${p.people.join(', ')}</div>` : ''}
        </div>
    `).join('');
}

// Render meetings
function renderMeetings(meetings) {
    if (meetings.length === 0) {
        detailItems.innerHTML = `<div class="brain-empty"><p>No meetings extracted yet</p></div>`;
        return;
    }
    detailItems.innerHTML = meetings.map(m => `
        <div class="detail-item profile-card">
            <div class="profile-header">
                <span class="profile-avatar">📅</span>
                <div class="profile-info">
                    <div class="profile-name">${escapeHtml(m.title || 'Meeting')}</div>
                    <div class="profile-role">${escapeHtml(m.when || 'Time TBD')}</div>
                </div>
            </div>
            ${m.agenda ? `<div class="profile-summary">${escapeHtml(m.agenda)}</div>` : ''}
            ${m.participants && m.participants.length ? `<div class="profile-team">👥 ${m.participants.join(', ')}</div>` : ''}
        </div>
    `).join('');
}

// Render generic items (for categories without special handling)
function renderGenericItems(items) {
    if (!items || items.length === 0) {
        detailItems.innerHTML = `<div class="brain-empty"><p>No items in this category yet</p></div>`;
        return;
    }
    detailItems.innerHTML = items.map(item => {
        const entities = (item.entities || [])
            .map(e => `<span class="entity-tag">${e.name}</span>`)
            .join('');
        const date = new Date(item.timestamp);
        const dateStr = date.toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        return `
            <div class="detail-item">
                <div class="detail-item-content">${escapeHtml(item.content)}</div>
                <div class="detail-item-meta"><span>${dateStr}</span></div>
                ${entities ? `<div class="detail-item-entities">${entities}</div>` : ''}
            </div>
        `;
    }).join('');
}

// Show categories grid (hide detail)
function showCategoriesGrid() {
    categoryDetail.classList.remove('active');
    categoriesGrid.style.display = '';
    brainStats.style.display = '';
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('peoples-agent-theme')) {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
    }
});
