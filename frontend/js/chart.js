
// stock_prediction_assistant/frontend/js/chart.js
document.addEventListener('DOMContentLoaded', function() {
    // 初始化Chart.js
    const ctx = document.getElementById('stock-chart').getContext('2d');
    let stockChart = new Chart(ctx, {
        type: 'candlestick',
        data: {
            datasets: [{
                label: 'K线图',
                data: [],
                color: {
                    up: '#10b981',
                    down: '#ef4444',
                    unchanged: '#9ca3af',
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                },
                y: {
                    ticks: {
                        callback: function(value) {
                            return '¥' + value.toFixed(2);
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const data = context.raw;
                            return [
                                '开盘: ' + data.o.toFixed(2),
                                '最高: ' + data.h.toFixed(2),
                                '最低: ' + data.l.toFixed(2),
                                '收盘: ' + data.c.toFixed(2)
                            ];
                        }
                    }
                }
            }
        }
    });

    // 从localStorage加载自选股
    function loadWatchlist() {
        const watchlist = JSON.parse(localStorage.getItem('watchlist')) || [];
        const watchlistEl = document.getElementById('watchlist');
        watchlistEl.innerHTML = '';
        
        watchlist.forEach(stock => {
            const li = document.createElement('li');
            li.className = 'flex justify-between items-center bg-gray-700 rounded-lg px-3 py-2 cursor-pointer hover:bg-gray-600 transition';
            li.innerHTML = `
                <span>${stock.name} (${stock.code})</span>
                <button class="remove-btn text-red-400 hover:text-red-300" data-code="${stock.code}">
                    <i class="fas fa-times"></i>
                </button>
            `;
            li.addEventListener('click', () => loadStockData(stock.code, stock.name));
            watchlistEl.appendChild(li);
        });

        // 添加删除按钮事件
        document.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                const code = this.getAttribute('data-code');
                removeFromWatchlist(code);
            });
        });
    }

    // 添加到自选股
    function addToWatchlist(code, name) {
        const watchlist = JSON.parse(localStorage.getItem('watchlist')) || [];
        if (!watchlist.some(stock => stock.code === code)) {
            watchlist.push({ code, name });
            localStorage.setItem('watchlist', JSON.stringify(watchlist));
            loadWatchlist();
        }
    }

    // 从自选股移除
    function removeFromWatchlist(code) {
        let watchlist = JSON.parse(localStorage.getItem('watchlist')) || [];
        watchlist = watchlist.filter(stock => stock.code !== code);
        localStorage.setItem('watchlist', JSON.stringify(watchlist));
        loadWatchlist();
    }

    // 加载股票数据（模拟）
    function loadStockData(code, name) {
        document.getElementById('stock-name').textContent = `${name} (${code})`;
        
        // 模拟数据
        const data = generateMockData(30);
        stockChart.data.datasets[0].data = data;
        stockChart.update();
    }

    // 生成模拟K线数据
    function generateMockData(days) {
        const data = [];
        let price = 100 + Math.random() * 20;
        
        for (let i = 0; i < days; i++) {
            const open = price;
            const close = open + (Math.random() - 0.5) * 10;
            const high = Math.max(open, close) + Math.random() * 5;
            const low = Math.min(open, close) - Math.random() * 5;
            const date = new Date();
            date.setDate(date.getDate() - (days - i));
            
            data.push({
                x: date,
                o: open,
                h: high,
                l: low,
                c: close
            });
            
            price = close;
        }
        
        return data;
    }

    // 搜索股票事件
    document.getElementById('search-btn').addEventListener('click', function() {
        const input = document.getElementById('stock-search').value.trim();
        if (input) {
            // 模拟搜索
            const code = input.match(/\d+/)?.[0] || '600000';
            const name = input.replace(code, '').trim() || '浦发银行';
            
            addToWatchlist(code, name);
            loadStockData(code, name);
            document.getElementById('stock-search').value = '';
        }
    });

    // 时间框架切换
    document.getElementById('timeframe').addEventListener('change', function() {
        const days = {
            '1d': 1,
            '1w': 7,
            '1m': 30,
            '3m': 90,
            '1y': 365
        }[this.value];
        
        if (days && stockChart.data.datasets[0].data.length > 0) {
            const code = document.getElementById('stock-name').textContent.match(/\((\d+)\)/)?.[1];
            const name = document.getElementById('stock-name').textContent.split('(')[0].trim();
            loadStockData(code, name);
        }
    });

    // 初始化加载
    loadWatchlist();
});
