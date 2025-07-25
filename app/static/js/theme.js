
// Professional Theme Toggle System
class ThemeManager {
    constructor() {
        this.init();
    }

    init() {
        // Load saved theme or default to dark
        const savedTheme = localStorage.getItem('theme') || 'dark';
        this.setTheme(savedTheme);
        
        // Bind event listeners
        this.bindEvents();
    }

    bindEvents() {
        const themeSwitch = document.querySelector('.switch-track');
        if (themeSwitch) {
            themeSwitch.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        
        // Update switch position
        const switchCircle = document.querySelector('.switch-circle');
        const sun = document.querySelector('.sun');
        const moon = document.querySelector('.moon');
        
        if (switchCircle && sun && moon) {
            if (theme === 'light') {
                switchCircle.style.transform = 'translateX(32px)';
                sun.style.opacity = '1';
                sun.style.color = 'var(--accent)';
                moon.style.opacity = '0.7';
                moon.style.color = 'var(--text-secondary)';
            } else {
                switchCircle.style.transform = 'translateX(0px)';
                moon.style.opacity = '1';
                moon.style.color = 'var(--accent)';
                sun.style.opacity = '0.7';
                sun.style.color = 'var(--text-secondary)';
            }
        }

        // Update charts if they exist
        this.updateCharts(theme);
        
        // Animate theme transition
        this.animateThemeTransition();
    }

    updateCharts(theme) {
        const charts = document.querySelectorAll('[id$="Chart"]');
        charts.forEach(chart => {
            if (chart.data && typeof chart.data === 'object') {
                Plotly.relayout(chart, {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { 
                        color: theme === 'dark' ? '#e0e0e0' : '#1a202c'
                    },
                    xaxis: {
                        gridcolor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                        zerolinecolor: theme === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'
                    },
                    yaxis: {
                        gridcolor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                        zerolinecolor: theme === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'
                    }
                });
            }
        });
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    animateThemeTransition() {
        // Add smooth transition class
        document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
        
        // Remove transition after animation
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }
}

// Professional Loading Animation System
class LoadingManager {
    static showButtonLoading(buttonSelector) {
        const button = document.querySelector(buttonSelector);
        if (!button) return;

        const btnText = button.querySelector('.btn-text');
        const btnLoader = button.querySelector('.btn-loader');
        
        if (btnText && btnLoader) {
            btnText.classList.add('d-none');
            btnLoader.classList.remove('d-none');
            button.disabled = true;
        }
    }

    static hideButtonLoading(buttonSelector) {
        const button = document.querySelector(buttonSelector);
        if (!button) return;

        const btnText = button.querySelector('.btn-text');
        const btnLoader = button.querySelector('.btn-loader');
        
        if (btnText && btnLoader) {
            btnText.classList.remove('d-none');
            btnLoader.classList.add('d-none');
            button.disabled = false;
        }
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme manager
    new ThemeManager();
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Global functions for backwards compatibility
function showLoadingMessage() {
    LoadingManager.showButtonLoading('.predict-btn');
}
