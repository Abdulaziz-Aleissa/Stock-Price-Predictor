
document.addEventListener('DOMContentLoaded', () => {
    const themeSwitch = document.querySelector('.theme-switch');
    const html = document.documentElement;
    
    // Check saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    html.setAttribute('data-theme', savedTheme);
    
    themeSwitch.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Update any charts if they exist
        const charts = document.querySelectorAll('[id$="Chart"]');
        charts.forEach(chart => {
            if (chart.data && typeof chart.data === 'object') {
                Plotly.relayout(chart, {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { 
                        color: newTheme === 'dark' ? '#e0e0e0' : '#1a202c'
                    },
                    xaxis: {
                        gridcolor: newTheme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                        zerolinecolor: newTheme === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'
                    },
                    yaxis: {
                        gridcolor: newTheme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                        zerolinecolor: newTheme === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'
                    }
                });
            }
        });
    });
});
