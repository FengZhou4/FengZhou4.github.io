
// stock_prediction_assistant/frontend/js/indicators.js
document.addEventListener('DOMContentLoaded', function() {
    // MACD指标计算
    function calculateMACD(data, shortPeriod = 12, longPeriod = 26, signalPeriod = 9) {
        if (data.length < longPeriod + signalPeriod) {
            console.warn('数据不足，无法计算MACD');
            return null;
        }

        // 计算EMA
        function calculateEMA(period, data, key = 'c') {
            const k = 2 / (period + 1);
            const ema = [];
            let sum = 0;
            
            // 计算SMA作为初始值
            for (let i = 0; i < period; i++) {
                sum += data[i][key];
            }
            ema[period - 1] = sum / period;
            
            // 计算后续EMA
            for (let i = period; i < data.length; i++) {
                ema[i] = (data[i][key] - ema[i - 1]) * k + ema[i - 1];
            }
            
            return ema;
        }

        const shortEMA = calculateEMA(shortPeriod, data);
        const longEMA = calculateEMA(longPeriod, data);
        const dif = [];
        const dea = [];
        const macd = [];

        // 计算DIF (差离值)
        for (let i = longPeriod - 1; i < data.length; i++) {
            dif[i] = shortEMA[i] - longEMA[i];
        }

        // 计算DEA (信号线)
        const signalEMA = calculateEMA(signalPeriod, dif.map((val, idx) => ({c: val})), 'c');
        for (let i = longPeriod + signalPeriod - 2; i < data.length; i++) {
            dea[i] = signalEMA[i];
        }

        // 计算MACD柱
        for (let i = longPeriod + signalPeriod - 2; i < data.length; i++) {
            macd[i] = (dif[i] - dea[i]) * 2;
        }

        return {
            dif,
            dea,
            macd,
            startIndex: longPeriod + signalPeriod - 2
        };
    }

    // KDJ指标计算
    function calculateKDJ(data, n = 9, m1 = 3, m2 = 3) {
        if (data.length < n + m1 + m2) {
            console.warn('数据不足，无法计算KDJ');
            return null;
        }

        const k = [];
        const d = [];
        const j = [];
        const rsv = [];

        for (let i = n - 1; i < data.length; i++) {
            // 计算RSV
            let highest = -Infinity;
            let lowest = Infinity;
            
            for (let j = i - n + 1; j <= i; j++) {
                highest = Math.max(highest, data[j].h);
                lowest = Math.min(lowest, data[j].l);
            }
            
            const currentClose = data[i].c;
            rsv[i] = (currentClose - lowest) / (highest - lowest) * 100;

            // 计算K值
            if (i === n - 1) {
                k[i] = 50; // 初始值
            } else {
                k[i] = (rsv[i] + (m1 - 1) * k[i - 1]) / m1;
            }

            // 计算D值
            if (i === n - 1) {
                d[i] = 50; // 初始值
            } else {
                d[i] = (k[i] + (m2 - 1) * d[i - 1]) / m2;
            }

            // 计算J值
            j[i] = 3 * k[i] - 2 * d[i];
        }

        return {
            k,
            d,
            j,
            startIndex: n - 1
        };
    }

    // 应用指标到图表
    function applyIndicatorsToChart(chart, indicators) {
        // 清除现有指标数据集
        chart.data.datasets = chart.data.datasets.filter(ds => ds.label === 'K线图');
        
        // 添加MACD指标
        if (indicators.macd) {
            chart.data.datasets.push({
                label: 'MACD',
                data: indicators.macd.macd.map((val, idx) => ({
                    x: chart.data.datasets[0].data[idx].x,
                    y: val
                })),
                type: 'bar',
                backgroundColor: function(context) {
                    return context.raw.y >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)';
                },
                borderColor: function(context) {
                    return context.raw.y >= 0 ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)';
                },
                borderWidth: 1,
                yAxisID: 'macd'
            });

            chart.data.datasets.push({
                label: 'DIF',
                data: indicators.macd.dif.map((val, idx) => ({
                    x: chart.data.datasets[0].data[idx].x,
                    y: val
                })),
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 2,
                pointRadius: 0,
                yAxisID: 'macd'
            });

            chart.data.datasets.push({
                label: 'DEA',
                data: indicators.macd.dea.map((val, idx) => ({
                    x: chart.data.datasets[0].data[idx].x,
                    y: val
                })),
                borderColor: 'rgba(245, 158, 11, 1)',
                borderWidth: 2,
                pointRadius: 0,
                yAxisID: 'macd'
            });
        }

        // 添加KDJ指标
        if (indicators.kdj) {
            chart.data.datasets.push({
                label: 'K',
                data: indicators.kdj.k.map((val, idx) => ({
                    x: chart.data.datasets[0].data[idx].x,
                    y: val
                })),
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1,
                pointRadius: 0,
                yAxisID: 'kdj'
            });

            chart.data.datasets.push({
                label: 'D',
                data: indicators.kdj.d.map((val, idx) => ({
                    x: chart.data.datasets[0].data[idx].x,
                    y: val
                })),
                borderColor: 'rgba(245, 158, 11, 1)',
                borderWidth: 1,
                pointRadius: 0,
                yAxisID: 'kdj'
            });

            chart.data.datasets.push({
                label: 'J',
                data: indicators.kdj.j.map((val, idx) => ({
                    x: chart.data.datasets[0].data[idx].x,
                    y: val
                })),
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 1,
                pointRadius: 0,
                yAxisID: 'kdj'
            });
        }

        // 更新图表
        chart.update();
    }

    // 应用指标按钮事件
    document.getElementById('apply-indicators').addEventListener('click', function() {
        const chart = Chart.getChart('stock-chart');
        if (!chart || chart.data.datasets[0].data.length === 0) {
            alert('请先加载股票数据');
            return;
        }

        const shortPeriod = parseInt(document.querySelector('#indicators-container input:nth-child(1)').value) || 12;
        const longPeriod = parseInt(document.querySelector('#indicators-container input:nth-child(2)').value) || 26;
        const signalPeriod = parseInt(document.querySelector('#indicators-container input:nth-child(3)').value) || 9;

        const kdjN = parseInt(document.querySelector('#indicators-container input:nth-child(4)').value) || 9;
        const kdjM1 = parseInt(document.querySelector('#indicators-container input:nth-child(5)').value) || 3;
        const kdjM2 = parseInt(document.querySelector('#indicators-container input:nth-child(6)').value) || 3;

        const data = chart.data.datasets[0].data;
        const indicators = {
            macd: calculateMACD(data, shortPeriod, longPeriod, signalPeriod),
            kdj: calculateKDJ(data, kdjN, kdjM1, kdjM2)
        };

        applyIndicatorsToChart(chart, indicators);
    });
});
