/**
 * Interactive Stock Dashboard JavaScript
 * Provides functionality for real-time stock analysis, charts, and data display
 */

class StockDashboard {
    constructor() {
        this.selectedTickers = new Set();
        this.chartData = {};
        this.refreshInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.loadDefaultStocks();
    }

    setupEventListeners() {
        // Stock search and add functionality
        const stockSearchInput = document.getElementById('stockSearchInput');
        const addStockBtn = document.getElementById('addStockBtn');
        
        if (stockSearchInput) {
            stockSearchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.addStock();
                }
            });
            
            // Real-time validation
            stockSearchInput.addEventListener('input', (e) => {
                this.validateTickerInput(e.target);
            });
        }
        
        if (addStockBtn) {
            addStockBtn.addEventListener('click', () => this.addStock());
        }

        // Refresh button
        const refreshBtn = document.getElementById('dashboardRefreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshAllData());
        }

        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('autoRefreshToggle');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                this.toggleAutoRefresh(e.target.checked);
            });
        }
    }

    validateTickerInput(input) {
        const symbol = input.value.trim().toUpperCase();
        const isValid = /^[A-Z]{1,5}$/.test(symbol);
        
        if (!isValid && symbol.length > 0) {
            input.classList.add('is-invalid');
            input.classList.remove('is-valid');
        } else {
            input.classList.remove('is-invalid');
            if (symbol.length > 0) {
                input.classList.add('is-valid');
            }
        }
        
        return isValid;
    }

    async addStock() {
        const input = document.getElementById('stockSearchInput');
        const symbol = input.value.trim().toUpperCase();
        
        if (!symbol || !this.validateTickerInput(input)) {
            this.showAlert('Please enter a valid stock symbol (1-5 letters)', 'warning');
            return;
        }

        if (this.selectedTickers.has(symbol)) {
            this.showAlert(`${symbol} is already in your dashboard`, 'info');
            return;
        }

        try {
            this.showLoading(true);
            
            // Add stock to dashboard
            this.selectedTickers.add(symbol);
            
            // Fetch and display stock data
            await this.loadStockData(symbol);
            
            // Clear input
            input.value = '';
            input.classList.remove('is-valid');
            
            this.showAlert(`${symbol} added to dashboard successfully!`, 'success');
            
        } catch (error) {
            this.selectedTickers.delete(symbol);
            this.showAlert(`Error adding ${symbol}: ${error.message}`, 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    async loadStockData(symbol) {
        try {
            // Fetch stock overview data
            const overviewResponse = await fetch(`/api/stock-overview/${symbol}`);
            if (!overviewResponse.ok) throw new Error('Failed to fetch stock data');
            const overviewData = await overviewResponse.json();

            // Fetch historical data for chart
            const historyResponse = await fetch(`/api/stock-history/${symbol}`);
            if (!historyResponse.ok) throw new Error('Failed to fetch historical data');
            const historyData = await historyResponse.json();

            // Update UI with stock data
            this.updateStockOverviewCard(symbol, overviewData);
            this.updateStockChart(symbol, historyData);
            
        } catch (error) {
            console.error(`Error loading data for ${symbol}:`, error);
            throw error;
        }
    }

    updateStockOverviewCard(symbol, data) {
        const container = document.getElementById('stockOverviewCards');
        
        const cardHtml = `
            <div class="col-md-6 col-lg-4 mb-4" id="stock-card-${symbol}">
                <div class="stock-overview-card">
                    <div class="stock-header">
                        <div class="stock-symbol">
                            <h4>${symbol}</h4>
                            <span class="company-name">${data.longName || symbol}</span>
                        </div>
                        <button class="btn btn-outline-danger btn-sm" onclick="dashboard.removeStock('${symbol}')">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    
                    <div class="stock-price">
                        <div class="current-price">$${data.regularMarketPrice?.toFixed(2) || 'N/A'}</div>
                        <div class="price-change ${data.regularMarketChange >= 0 ? 'positive' : 'negative'}">
                            ${data.regularMarketChange >= 0 ? '+' : ''}${data.regularMarketChange?.toFixed(2) || '0.00'} 
                            (${data.regularMarketChangePercent >= 0 ? '+' : ''}${data.regularMarketChangePercent?.toFixed(2) || '0.00'}%)
                        </div>
                    </div>
                    
                    <div class="stock-metrics">
                        <div class="metric">
                            <span class="metric-label">Market Cap</span>
                            <span class="metric-value">${this.formatMarketCap(data.marketCap)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Volume</span>
                            <span class="metric-value">${this.formatVolume(data.volume)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">P/E Ratio</span>
                            <span class="metric-value">${data.forwardPE?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">52W Range</span>
                            <span class="metric-value">$${data.fiftyTwoWeekLow?.toFixed(2) || 'N/A'} - $${data.fiftyTwoWeekHigh?.toFixed(2) || 'N/A'}</span>
                        </div>
                    </div>
                    
                    <div class="stock-actions">
                        <button class="btn btn-primary btn-sm" onclick="dashboard.showStockDetails('${symbol}')">
                            <i class="fas fa-chart-line"></i> View Chart
                        </button>
                        <button class="btn btn-outline-primary btn-sm" onclick="dashboard.addToWatchlist('${symbol}')">
                            <i class="fas fa-eye"></i> Watch
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', cardHtml);
    }

    updateStockChart(symbol, historyData) {
        const chartContainer = document.getElementById(`chart-${symbol}`);
        
        if (!chartContainer) {
            // Create chart container if it doesn't exist
            const chartsSection = document.getElementById('stockChartsSection');
            const chartHtml = `
                <div class="col-md-12 mb-4" id="chart-container-${symbol}">
                    <div class="chart-card">
                        <div class="chart-header">
                            <h5>${symbol} - Price Chart</h5>
                            <div class="chart-controls">
                                <button class="btn btn-outline-secondary btn-sm" onclick="dashboard.updateChartPeriod('${symbol}', '1M')">1M</button>
                                <button class="btn btn-outline-secondary btn-sm" onclick="dashboard.updateChartPeriod('${symbol}', '3M')">3M</button>
                                <button class="btn btn-outline-secondary btn-sm" onclick="dashboard.updateChartPeriod('${symbol}', '6M')">6M</button>
                                <button class="btn btn-outline-secondary btn-sm" onclick="dashboard.updateChartPeriod('${symbol}', '1Y')">1Y</button>
                            </div>
                        </div>
                        <div id="chart-${symbol}" class="stock-chart"></div>
                    </div>
                </div>
            `;
            chartsSection.insertAdjacentHTML('beforeend', chartHtml);
        }

        this.renderChart(symbol, historyData);
    }

    renderChart(symbol, data) {
        const chartDiv = document.getElementById(`chart-${symbol}`);
        
        const trace1 = {
            x: data.dates,
            y: data.prices,
            type: 'scatter',
            mode: 'lines',
            name: 'Price',
            line: {
                color: '#3b82f6',
                width: 2
            }
        };

        const trace2 = {
            x: data.dates,
            y: data.sma20,
            type: 'scatter',
            mode: 'lines',
            name: 'SMA 20',
            line: {
                color: '#f59e0b',
                width: 1,
                dash: 'dot'
            }
        };

        const trace3 = {
            x: data.dates,
            y: data.sma50,
            type: 'scatter',
            mode: 'lines',
            name: 'SMA 50',
            line: {
                color: '#ef4444',
                width: 1,
                dash: 'dot'
            }
        };

        const layout = {
            title: `${symbol} Stock Price`,
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' },
            margin: { l: 50, r: 50, t: 50, b: 50 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { 
                color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim()
            },
            showlegend: true,
            legend: {
                x: 0,
                y: 1,
                bgcolor: 'rgba(0,0,0,0)'
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
        };

        Plotly.newPlot(chartDiv, [trace1, trace2, trace3], layout, config);
    }

    removeStock(symbol) {
        this.selectedTickers.delete(symbol);
        
        // Remove stock card
        const stockCard = document.getElementById(`stock-card-${symbol}`);
        if (stockCard) {
            stockCard.remove();
        }
        
        // Remove chart
        const chartContainer = document.getElementById(`chart-container-${symbol}`);
        if (chartContainer) {
            chartContainer.remove();
        }
        
        this.showAlert(`${symbol} removed from dashboard`, 'info');
    }

    showStockDetails(symbol) {
        // Toggle chart visibility or scroll to chart
        const chartContainer = document.getElementById(`chart-container-${symbol}`);
        if (chartContainer) {
            chartContainer.scrollIntoView({ behavior: 'smooth' });
        }
    }

    async addToWatchlist(symbol) {
        try {
            const response = await fetch('/add_to_watchlist', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `symbol=${symbol}&target_price=0`
            });
            
            if (response.ok) {
                this.showAlert(`${symbol} added to watchlist`, 'success');
            } else {
                throw new Error('Failed to add to watchlist');
            }
        } catch (error) {
            this.showAlert(`Error adding ${symbol} to watchlist`, 'danger');
        }
    }

    formatMarketCap(marketCap) {
        if (!marketCap) return 'N/A';
        
        if (marketCap >= 1e12) {
            return `$${(marketCap / 1e12).toFixed(2)}T`;
        } else if (marketCap >= 1e9) {
            return `$${(marketCap / 1e9).toFixed(2)}B`;
        } else if (marketCap >= 1e6) {
            return `$${(marketCap / 1e6).toFixed(2)}M`;
        } else {
            return `$${marketCap.toLocaleString()}`;
        }
    }

    formatVolume(volume) {
        if (!volume) return 'N/A';
        
        if (volume >= 1e6) {
            return `${(volume / 1e6).toFixed(2)}M`;
        } else if (volume >= 1e3) {
            return `${(volume / 1e3).toFixed(2)}K`;
        } else {
            return volume.toLocaleString();
        }
    }

    async refreshAllData() {
        if (this.selectedTickers.size === 0) {
            this.showAlert('No stocks to refresh', 'info');
            return;
        }

        try {
            this.showLoading(true);
            
            for (const symbol of this.selectedTickers) {
                await this.loadStockData(symbol);
            }
            
            this.showAlert('Dashboard refreshed successfully', 'success');
            
        } catch (error) {
            this.showAlert('Error refreshing dashboard', 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    toggleAutoRefresh(enabled) {
        if (enabled) {
            this.refreshInterval = setInterval(() => {
                this.refreshAllData();
            }, 60000); // Refresh every minute
            this.showAlert('Auto-refresh enabled (1 minute intervals)', 'success');
        } else {
            if (this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
            this.showAlert('Auto-refresh disabled', 'info');
        }
    }

    loadDefaultStocks() {
        // Load some popular stocks by default
        const defaultStocks = ['AAPL', 'GOOGL', 'MSFT'];
        defaultStocks.forEach(symbol => {
            this.selectedTickers.add(symbol);
            this.loadStockData(symbol).catch(console.error);
        });
    }

    initializeCharts() {
        // Initialize any default charts or chart containers
        const chartsSection = document.getElementById('stockChartsSection');
        if (chartsSection && chartsSection.children.length === 0) {
            chartsSection.innerHTML = '<div class="text-center text-muted py-4"><i class="fas fa-chart-line fa-2x mb-2"></i><br>Add stocks to see interactive charts</div>';
        }
    }

    showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('dashboardAlerts');
        if (!alertsContainer) return;

        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        alertsContainer.insertAdjacentHTML('afterbegin', alertHtml);

        // Auto-remove alert after 5 seconds
        setTimeout(() => {
            const alert = alertsContainer.firstElementChild;
            if (alert && alert.classList.contains('alert')) {
                alert.remove();
            }
        }, 5000);
    }

    showLoading(show) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = show ? 'block' : 'none';
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new StockDashboard();
});