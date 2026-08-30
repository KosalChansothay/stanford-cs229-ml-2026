/* ==========================================================================
   CS336 Lecture 9 interactive Plotly charts — Scaling Laws (Foundations).
   Chart 1: IsoFLOP Loss Contours & Optimal Frontier (Chinchilla 20:1 Ratio)
   Chart 2: Kaplan vs Chinchilla Compute Allocation (Parameters vs Tokens)
   ========================================================================== */

(function () {
    'use strict';

    function drawIsoflop(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Model loss L(N, D) = E + A / N^alpha + B / D^beta
        // Chinchilla parameters: E = 1.69, A = 406.4, alpha = 0.34, B = 410.7, beta = 0.28
        // Compute budget C = 6 * N * D => D = C / (6 * N)
        var budgets = [1e19, 1e20, 1e21, 1e22, 1e23];
        var budgetLabels = ['1e19 FLOPs', '1e20 FLOPs', '1e21 FLOPs', '1e22 FLOPs', '1e23 FLOPs'];
        var colors = ['#519db8', '#286b82', '#3d8b57', '#e69f00', '#b84a39'];

        var traces = [];
        var optimalN = [];
        var optimalLoss = [];

        budgets.forEach(function (C, idx) {
            var nValues = [];
            var lossValues = [];
            // Sample model parameter sizes N
            for (var p = 7; p <= 11.5; p += 0.1) {
                var N = Math.pow(10, p);
                var D = C / (6 * N);
                if (D > 1e8) {
                    var loss = 1.69 + 406.4 / Math.pow(N, 0.34) + 410.7 / Math.pow(D, 0.28);
                    nValues.push(N);
                    lossValues.push(loss);
                }
            }

            // Find minimum loss for this compute budget
            var minLoss = Infinity, bestN = nValues[0];
            for (var i = 0; i < lossValues.length; i++) {
                if (lossValues[i] < minLoss) { minLoss = lossValues[i]; bestN = nValues[i]; }
            }
            optimalN.push(bestN);
            optimalLoss.push(minLoss);

            traces.push({
                x: nValues,
                y: lossValues,
                mode: 'lines',
                name: 'IsoFLOP: ' + budgetLabels[idx],
                line: { color: colors[idx], width: 2.2 }
            });
        });

        // Add optimal frontier trace
        traces.push({
            x: optimalN,
            y: optimalLoss,
            mode: 'lines+markers',
            name: '<b>Chinchilla Optimal Frontier (20:1 Ratio)</b>',
            line: { color: '#1f3a52', width: 3.5, dash: 'dot' },
            marker: { size: 9, color: '#1f3a52', symbol: 'diamond' }
        });

        var layout = {
            title: { text: '<b>Chinchilla IsoFLOP Curves & Optimal Model Size Frontier</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Non-Embedding Parameters (N)', gridcolor: '#e5e5e5', type: 'log' },
            yaxis: { title: 'Validation Loss L(N, D)', gridcolor: '#e5e5e5' },
            height: 460,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.14, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, traces, layout, { displayModeBar: false });
    }

    function drawKaplanVsChinchilla(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var computes = [1e19, 1e20, 1e21, 1e22, 1e23, 1e24];
        var xLabels = ['1e19', '1e20', '1e21', '1e22', '1e23', '1e24'];

        // Kaplan: N ~ C^0.73, D ~ C^0.27
        // Chinchilla: N ~ C^0.50, D ~ C^0.50
        var chinchillaParamsB = [0.29, 0.91, 2.88, 9.12, 28.8, 91.2]; // Billion params
        var kaplanParamsB = [0.08, 0.42, 2.26, 12.1, 64.9, 348.0]; // Over-allocates parameters

        var data = [
            {
                x: xLabels,
                y: chinchillaParamsB,
                mode: 'lines+markers',
                name: 'Chinchilla Optimal Allocation (N ∝ C^0.50)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 8 }
            },
            {
                x: xLabels,
                y: kaplanParamsB,
                mode: 'lines+markers',
                name: 'Kaplan Suboptimal Allocation (N ∝ C^0.73)',
                line: { color: '#b84a39', width: 3, dash: 'dash' },
                marker: { size: 8 }
            }
        ];

        var layout = {
            title: { text: '<b>Parameter Allocation: Chinchilla (Balanced) vs Kaplan (Parameter-Heavy)</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Training Compute Budget (FLOPs)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Optimal Parameter Count (Billions)', gridcolor: '#e5e5e5', type: 'log' },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture9Charts = function () {
        drawIsoflop('plotly-cs336-9-isoflop');
        drawKaplanVsChinchilla('plotly-cs336-9-kaplan-vs-chinchilla');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture9Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-9-isoflop')) {
        window.renderCs336Lecture9Charts();
    }
})();
