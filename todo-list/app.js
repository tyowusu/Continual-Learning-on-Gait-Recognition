/**
 * Todo List Application with Local Storage
 * 
 * Features:
 * - Add, edit, delete tasks
 * - Priority levels (Low, Medium, High)
 * - Filter by status and priority
 * - Sort by multiple criteria
 * - Statistics dashboard
 * - Export/Import functionality
 * - Persistent local storage
 */

class TodoApp {
    constructor() {
        this.todos = [];
        this.currentFilter = 'all';
        this.currentSort = 'date-desc';
        this.storageKey = 'todos_data';
        
        this.initializeDOM();
        this.loadTodos();
        this.attachEventListeners();
        this.render();
    }

    /**
     * Initialize DOM element references
     */
    initializeDOM() {
        this.todoInput = document.getElementById('todoInput');
        this.prioritySelect = document.getElementById('prioritySelect');
        this.addBtn = document.getElementById('addBtn');
        this.todoList = document.getElementById('todoList');
        this.filterBtn = document.getElementById('filterBtn');
        this.filterPanel = document.getElementById('filterPanel');
        this.statsBtn = document.getElementById('statsBtn');
        this.statsPanel = document.getElementById('statsPanel');
        this.sortSelect = document.getElementById('sortSelect');
        this.clearCompletedBtn = document.getElementById('clearCompletedBtn');
        this.clearAllBtn = document.getElementById('clearAllBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.importBtn = document.getElementById('importBtn');
        this.fileInput = document.getElementById('fileInput');
    }

    /**
     * Attach event listeners to DOM elements
     */
    attachEventListeners() {
        this.addBtn.addEventListener('click', () => this.addTodo());
        this.todoInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addTodo();
        });

        // Filter buttons
        this.filterBtn.addEventListener('click', () => this.toggleFilterPanel());
        document.querySelectorAll('.filter-option').forEach(btn => {
            btn.addEventListener('click', (e) => this.setFilter(e.target.dataset.filter));
        });

        // Stats button
        this.statsBtn.addEventListener('click', () => this.toggleStatsPanel());

        // Sort
        this.sortSelect.addEventListener('change', (e) => {
            this.currentSort = e.target.value;
            this.render();
        });

        // Action buttons
        this.clearCompletedBtn.addEventListener('click', () => this.clearCompleted());
        this.clearAllBtn.addEventListener('click', () => this.clearAll());
        this.exportBtn.addEventListener('click', () => this.exportData());
        this.importBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.importData(e));

        // Close panels when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.filterBtn.contains(e.target) && !this.filterPanel.contains(e.target)) {
                this.filterPanel.classList.add('hidden');
            }
            if (!this.statsBtn.contains(e.target) && !this.statsPanel.contains(e.target)) {
                this.statsPanel.classList.add('hidden');
            }
        });
    }

    /**
     * Add a new todo
     */
    addTodo() {
        const text = this.todoInput.value.trim();
        if (!text) {
            alert('Please enter a task');
            return;
        }

        const todo = {
            id: Date.now(),
            text: text,
            priority: this.prioritySelect.value,
            completed: false,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        this.todos.unshift(todo);
        this.saveTodos();
        this.todoInput.value = '';
        this.todoInput.focus();
        this.render();
    }

    /**
     * Toggle todo completion status
     */
    toggleTodo(id) {
        const todo = this.todos.find(t => t.id === id);
        if (todo) {
            todo.completed = !todo.completed;
            todo.updatedAt = new Date().toISOString();
            this.saveTodos();
            this.render();
        }
    }

    /**
     * Edit a todo
     */
    editTodo(id) {
        const todo = this.todos.find(t => t.id === id);
        if (!todo) return;

        const newText = prompt('Edit task:', todo.text);
        if (newText && newText.trim()) {
            todo.text = newText.trim();
            todo.updatedAt = new Date().toISOString();
            this.saveTodos();
            this.render();
        }
    }

    /**
     * Delete a todo
     */
    deleteTodo(id) {
        if (confirm('Are you sure you want to delete this task?')) {
            this.todos = this.todos.filter(t => t.id !== id);
            this.saveTodos();
            this.render();
        }
    }

    /**
     * Toggle filter panel visibility
     */
    toggleFilterPanel() {
        this.filterPanel.classList.toggle('hidden');
        this.statsPanel.classList.add('hidden');
    }

    /**
     * Set current filter
     */
    setFilter(filter) {
        this.currentFilter = filter;
        document.querySelectorAll('.filter-option').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-filter="${filter}"]`).classList.add('active');
        this.filterPanel.classList.add('hidden');
        this.render();
    }

    /**
     * Toggle stats panel visibility
     */
    toggleStatsPanel() {
        this.statsPanel.classList.toggle('hidden');
        this.filterPanel.classList.add('hidden');
        this.updateStats();
    }

    /**
     * Filter todos based on current filter
     */
    filterTodos(todos) {
        switch (this.currentFilter) {
            case 'active':
                return todos.filter(t => !t.completed);
            case 'completed':
                return todos.filter(t => t.completed);
            case 'high':
                return todos.filter(t => t.priority === 'high');
            case 'medium':
                return todos.filter(t => t.priority === 'medium');
            case 'low':
                return todos.filter(t => t.priority === 'low');
            default:
                return todos;
        }
    }

    /**
     * Sort todos based on current sort option
     */
    sortTodos(todos) {
        const sorted = [...todos];
        
        switch (this.currentSort) {
            case 'date-asc':
                sorted.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
                break;
            case 'date-desc':
                sorted.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
                break;
            case 'priority':
                const priorityOrder = { high: 1, medium: 2, low: 3 };
                sorted.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);
                break;
            case 'alpha':
                sorted.sort((a, b) => a.text.localeCompare(b.text));
                break;
        }
        
        return sorted;
    }

    /**
     * Format date for display
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);

        if (date.toDateString() === today.toDateString()) {
            return 'Today';
        } else if (date.toDateString() === yesterday.toDateString()) {
            return 'Yesterday';
        } else {
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        }
    }

    /**
     * Render the todo list
     */
    render() {
        const filtered = this.filterTodos(this.todos);
        const sorted = this.sortTodos(filtered);

        if (sorted.length === 0) {
            this.todoList.innerHTML = `
                <div class="empty-state">
                    <p class="empty-icon">📝</p>
                    <p class="empty-message">No tasks to show. Keep up the great work!</p>
                </div>
            `;
            return;
        }

        this.todoList.innerHTML = sorted.map(todo => `
            <div class="todo-item ${todo.completed ? 'completed' : ''}">
                <input 
                    type="checkbox" 
                    class="checkbox" 
                    ${todo.completed ? 'checked' : ''}
                    onchange="app.toggleTodo(${todo.id})"
                >
                <div class="todo-content">
                    <span class="priority-badge ${todo.priority}">${todo.priority}</span>
                    <span class="todo-text">${this.escapeHtml(todo.text)}</span>
                </div>
                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
                    <div class="todo-meta">
                        <span class="todo-date">📅 ${this.formatDate(todo.createdAt)}</span>
                    </div>
                    <div class="todo-actions">
                        <button class="todo-btn edit-btn" onclick="app.editTodo(${todo.id})">Edit</button>
                        <button class="todo-btn delete-btn" onclick="app.deleteTodo(${todo.id})">Delete</button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    /**
     * Update statistics
     */
    updateStats() {
        const total = this.todos.length;
        const completed = this.todos.filter(t => t.completed).length;
        const remaining = total - completed;
        const rate = total === 0 ? 0 : Math.round((completed / total) * 100);

        document.getElementById('totalTasks').textContent = total;
        document.getElementById('completedTasks').textContent = completed;
        document.getElementById('remainingTasks').textContent = remaining;
        document.getElementById('completionRate').textContent = rate + '%';
    }

    /**
     * Clear completed todos
     */
    clearCompleted() {
        if (confirm('Delete all completed tasks?')) {
            this.todos = this.todos.filter(t => !t.completed);
            this.saveTodos();
            this.render();
        }
    }

    /**
     * Clear all todos
     */
    clearAll() {
        if (confirm('Delete ALL tasks? This cannot be undone!')) {
            this.todos = [];
            this.saveTodos();
            this.render();
        }
    }

    /**
     * Export todos as JSON
     */
    exportData() {
        const data = JSON.stringify(this.todos, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `todos_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    /**
     * Import todos from JSON file
     */
    importData(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const imported = JSON.parse(e.target.result);
                if (!Array.isArray(imported)) {
                    throw new Error('Invalid format');
                }
                if (confirm(`Import ${imported.length} tasks? This will merge with existing tasks.`)) {
                    this.todos = [...this.todos, ...imported];
                    this.saveTodos();
                    this.render();
                    alert('Tasks imported successfully!');
                }
            } catch (error) {
                alert('Error importing file: ' + error.message);
            }
        };
        reader.readAsText(file);
        this.fileInput.value = '';
    }

    /**
     * Save todos to local storage
     */
    saveTodos() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.todos));
        } catch (error) {
            console.error('Error saving to localStorage:', error);
            alert('Error saving tasks. Local storage may be full.');
        }
    }

    /**
     * Load todos from local storage
     */
    loadTodos() {
        try {
            const stored = localStorage.getItem(this.storageKey);
            if (stored) {
                this.todos = JSON.parse(stored);
            }
        } catch (error) {
            console.error('Error loading from localStorage:', error);
            this.todos = [];
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new TodoApp();
});
