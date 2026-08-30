/* ==========================================================================
   CS336 Lecture 10 interactive Plotly charts — Inference (Execution & Optimization).
   Chart 1: Prefill (Compute-Bound) vs Decode (Memory-Bound) Latency & MFU
   Chart 2: Speculative Decoding Speedup vs Token Acceptance Rate alpha
   ========================================================================== */

(function () {
    'use strict';

    function drawPrefillDecode(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var batchSizes = [1, 2, 4, 8, 16, 32, 64, 128];
        var xLabels = batchSizes.map(function (b) { return 'Batch ' + b; });

        // Decode per-token latency (ms): bounded by reading model weights from HBM (flat until compute saturates)
        var decodeLatency = [18.2, 18.5, 18.9, 19.8, 22.4, 28.5, 42.0, 75.0];
        // Decode throughput (tokens/sec): scales linearly with batch size while memory-bound
        var decodeThroughput = batchSizes.map(function (b, i) {
            return (b * 1000) / decodeLatency[i];
        });

        var data = [
            {
                x: xLabels,
                y: decodeLatency,
                name: 'Per-Step Decoding Latency (ms)',
                type: 'bar',
                yaxis: 'y1',
                marker: { color: '#b84a39' }
            },
            {
                x: xLabels,
                y: decodeThroughput,
                name: 'Aggregate Serving Throughput (tok/s)',
                type: 'scatter',
                mode: 'lines+markers',
                yaxis: 'y2',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>LLM Serving: Memory-Bound Decoding Latency & Throughput vs Batch Size</b>', x: 0.5, font: { size: 15 } },
            xaxis: { gridcolor: '#e5e5e5' },
            yaxis: { title: 'Latency per Token Step (ms)', titlefont: { color: '#b84a39' }, tickfont: { color: '#b84a39' }, gridcolor: '#e5e5e5' },
            yaxis2: { title: 'Aggregate Throughput (Tokens / sec)', titlefont: { color: '#286b82' }, tickfont: { color: '#286b82' }, overlaying: 'y', side: 'right' },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 65 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawSpeculativeDecoding(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var alphas = [];
        for (var a = 0; a <= 0.95; a += 0.05) alphas.push(a);

        // Speedup = (1 - alpha^(gamma + 1)) / ((1 - alpha) * (1 + c * gamma))
        // where gamma = lookahead tokens, c = draft model cost relative to target (e.g. c = 0.05)
        var c = 0.05;
        var gammas = [3, 5, 7];
        var colors = ['#519db8', '#286b82', '#3d8b57'];

        var traces = gammas.map(function (gamma, idx) {
            var speedups = alphas.map(function (alpha) {
                if (alpha === 1) return (gamma + 1) / (1 + c * gamma);
                var expectedAccepted = (1 - Math.pow(alpha, gamma + 1)) / (1 - alpha);
                return expectedAccepted / (1 + c * gamma);
            });
            return {
                x: alphas,
                y: speedups,
                mode: 'lines+markers',
                name: 'Lookahead γ = ' + gamma + ' Draft Tokens',
                line: { color: colors[idx], width: 3 },
                marker: { size: 6 }
            };
        });

        // Baseline (1.0x)
        traces.push({
            x: [0, 0.95],
            y: [1, 1],
            mode: 'lines',
            name: 'Standard Autoregressive Baseline (1.0x)',
            line: { color: '#999', width: 2, dash: 'dash' }
        });

        var layout = {
            title: { text: '<b>Speculative Decoding Speedup vs Draft Token Acceptance Rate (α)</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Draft Model Acceptance Probability (α)', gridcolor: '#e5e5e5', range: [0, 1] },
            yaxis: { title: 'Inference Speedup Factor', gridcolor: '#e5e5e5', range: [0.8, 3.5] },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, traces, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture10Charts = function () {
        drawPrefillDecode('plotly-cs336-10-prefill-decode');
        drawSpeculativeDecoding('plotly-cs336-10-speculative-decoding');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture10Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-10-prefill-decode')) {
        window.renderCs336Lecture10Charts();
    }
})();
