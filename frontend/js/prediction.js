
// stock_prediction_assistant/frontend/js/prediction.js
document.addEventListener('DOMContentLoaded', function() {
    // 获取预测按钮和结果显示区域
    const runPredictionBtn = document.getElementById('run-prediction');
    const predictionResult = document.getElementById('prediction-result');
    
    // 预测按钮点击事件
    runPredictionBtn.addEventListener('click', function() {
        const chart = Chart.getChart('stock-chart');
        if (!chart || chart.data.datasets[0].data.length === 0) {
            alert('请先加载股票数据');
            return;
        }
        
        // 显示加载状态
        predictionResult.innerHTML = `
            <div class="flex flex-col items-center">
                <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
                <p>正在分析数据...</p>
            </div>
        `;
        
        // 模拟异步预测过程
        setTimeout(() => {
            const prediction = analyzeStockData(chart.data.datasets[0].data);
            displayPredictionResult(prediction);
        }, 1500);
    });
    
    // 分析股票数据
    function analyzeStockData(data) {
        if (data.length < 20) {
            return {
                confidence: 0,
                trend: 'neutral',
                suggestion: '数据不足，无法做出可靠预测'
            };
        }
        
        // 计算短期(5日)和长期(20日)移动平均线
        const shortMA = calculateMovingAverage(data, 5);
        const longMA = calculateMovingAverage(data, 20);
        
        // 计算趋势强度
        const lastShort = shortMA[shortMA.length - 1];
        const lastLong = longMA[longMA.length - 1];
        const prevShort = shortMA[shortMA.length - 2];
        const prevLong = longMA[longMA.length - 2];
        
        // 判断趋势
        let trend, confidence;
        if (lastShort > lastLong && prevShort <= prevLong) {
            // 金叉
            trend = 'up';
            confidence = Math.min(0.8, (lastShort - lastLong) / lastLong * 10);
        } else if (lastShort < lastLong && prevShort >= prevLong) {
            // 死叉
            trend = 'down';
            confidence = Math.min(0.8, (lastLong - lastShort) / lastShort * 10);
        } else if (lastShort > lastLong) {
            // 上涨趋势
            trend = 'up';
            confidence = Math.min(0.6, (lastShort - lastLong) / lastLong * 5);
        } else if (lastShort < lastLong) {
            // 下跌趋势
            trend = 'down';
            confidence = Math.min(0.6, (lastLong - lastShort) / lastShort * 5);
        } else {
            // 盘整
            trend = 'neutral';
            confidence = 0.3;
        }
        
        // 生成建议
        let suggestion;
        if (trend === 'up' && confidence > 0.5) {
            suggestion = '强烈建议买入';
        } else if (trend === 'up') {
            suggestion = '可以考虑买入';
        } else if (trend === 'down' && confidence > 0.5) {
            suggestion = '强烈建议卖出';
        } else if (trend === 'down') {
            suggestion = '可以考虑卖出';
        } else {
            suggestion = '建议观望';
        }
        
        return {
            confidence: Math.round(confidence * 100),
            trend,
            suggestion
        };
    }
    
    // 计算移动平均线
    function calculateMovingAverage(data, period) {
        const ma = [];
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = i - period + 1; j <= i; j++) {
                sum += data[j].c;
            }
            ma.push(sum / period);
        }
        return ma;
    }
    
    // 显示预测结果
    function displayPredictionResult(result) {
        let trendIcon, trendColor;
        switch(result.trend) {
            case 'up':
                trendIcon = '📈';
                trendColor = 'text-green-500';
                break;
            case 'down':
                trendIcon = '📉';
                trendColor = 'text-red-500';
                break;
            default:
                trendIcon = '➖';
                trendColor = 'text-gray-500';
        }
        
        predictionResult.innerHTML = `
            <div class="text-center">
                <div class="text-4xl mb-2 ${trendColor}">${trendIcon}</div>
                <h3 class="text-xl font-semibold mb-2">${result.suggestion}</h3>
                <div class="flex justify-center items-center mb-4">
                    <span class="mr-2">置信度:</span>
                    <div class="w-32 bg-gray-700 rounded-full h-4">
                        <div class="bg-blue-500 h-4 rounded-full" style="width: ${result.confidence}%"></div>
                    </div>
                    <span class="ml-2">${result.confidence}%</span>
                </div>
                <p class="text-sm text-gray-400">基于移动平均线交叉分析</p>
            </div>
        `;
    }
    
    // 添加开发中提示
    runPredictionBtn.addEventListener('click', function() {
        if (Math.random() < 0.3) {
            setTimeout(() => {
                alert('高级预测功能正在开发中，敬请期待！');
            }, 100);
        }
    });
});
