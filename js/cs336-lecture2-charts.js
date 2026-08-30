/* ==========================================================================
   CS336 Lecture 2 interactive Plotly charts — PyTorch & Resource Accounting.
   Chart 1: NVIDIA H100 Roofline Analysis (log-log Arithmetic Intensity vs FLOPs/s)
   Chart 2: Memory Footprint Breakdown (Weights, Gradients, Optim, Activations)
   ========================================================================== */

(function () {
    'use strict';

    function drawRoofline(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var peakFlops = 989.5; // TFLOPs/s (Dense BF16 on H100)
        var memBandwidth = 3.35; // TB/s (HBM3 on H100)
        var kneeIntensity = peakFlops / memBandwidth; // ~295.37 FLOPs/byte

        // Generate roofline points
        var intensities = [];
        for (var p = -1; p <= 4; p += 0.1) {
            intensities.push(Math.pow(10, p));
        }

        var rooflineFlops = intensities.map(function (I) {
            return Math.min(peakFlops, memBandwidth * I);
        });

        // Operators to plot
        var ops = [
            { name: 'LayerNorm / RMSNorm', I: 0.5, type: 'Memory-Bound', color: '#b84a39' },
            { name: 'GELU / SwiGLU', I: 0.5, type: 'Memory-Bound', color: '#b84a39' },
            { name: 'Softmax', I: 0.67, type: 'Memory-Bound', color: '#b84a39' },
            { name: 'MatMul (N=128)', I: 128 / 3, type: 'Memory-Bound', color: '#e69f00' },
            { name: 'MatMul (N=512)', I: 512 / 3, type: 'Memory-Bound', color: '#e69f00' },
            { name: 'MatMul (N=2048)', I: 2048 / 3, type: 'Compute-Bound', color: '#286b82' },
            { name: 'MatMul (N=8192)', I: 8192 / 3, type: 'Compute-Bound', color: '#286b82' }
        ];

        var opX = ops.map(function (o) { return o.I; });
        var opY = ops.map(function (o) { return Math.min(peakFlops, memBandwidth * o.I); });
        var opLabels = ops.map(function (o) { return o.name + '<br>Intensity: ' + o.I.toFixed(1) + ' FLOP/B<br>Attainable: ' + Math.min(peakFlops, memBandwidth * o.I).toFixed(1) + ' TFLOPs'; });
        var opColors = ops.map(function (o) { return o.color; });

        var data = [
            {
                x: intensities,
                y: rooflineFlops,
                mode: 'lines',
                name: 'H100 Roofline Limit',
                line: { color: '#1f3a52', width: 3.5 },
                hoverinfo: 'none'
            },
            {
                x: opX,
                y: opY,
                mode: 'markers+text',
                name: 'Operators',
                text: ops.map(function (o) { return o.name; }),
                textposition: ['bottom right', 'top left', 'top right', 'top left', 'top right', 'bottom right', 'bottom left'],
                textfont: { size: 11, color: '#333' },
                marker: { size: 10, color: opColors, symbol: 'circle' },
                hovertemplate: '%{customdata}<extra></extra>',
                customdata: opLabels
            }
        ];

        var layout = {
            title: { text: '<b>NVIDIA H100 Roofline Analysis (Dense BF16)</b>', x: 0.5, font: { size: 15 } },
            xaxis: {
                title: 'Arithmetic Intensity (FLOPs / Byte transferred)',
                type: 'log',
                gridcolor: '#e5e5e5',
                range: [-0.5, 3.8]
            },
            yaxis: {
                title: 'Attainable Performance (TFLOPs / sec)',
                type: 'log',
                gridcolor: '#e5e5e5',
                range: [0, 3.2]
            },
            annotations: [
                {
                    x: Math.log10(2),
                    y: Math.log10(15),
                    xref: 'x',
                    yref: 'y',
                    text: '<b>Memory-Bound Regime</b><br>(Latency bound by HBM bandwidth)',
                    showarrow: false,
                    font: { size: 11, color: '#b84a39' },
                    align: 'center'
                },
                {
                    x: Math.log10(1500),
                    y: Math.log10(600),
                    xref: 'x',
                    yref: 'y',
                    text: '<b>Compute-Bound Regime</b><br>(Peak 989.5 TFLOPs/s)',
                    showarrow: false,
                    font: { size: 11, color: '#286b82' },
                    align: 'center'
                }
            ],
            height: 460,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 60, l: 65, r: 35 },
            showlegend: false
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawMemoryBreakdown(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Model: 7B parameters (N = 7e9)
        // FP16/BF16 weights: 2N bytes = 14 GB
        // Gradients: 2N bytes = 14 GB
        // AdamW states: 8N bytes (FP32 m, v) + 4N (FP32 master weights) = 12N bytes = 84 GB (or 8N = 56 GB)
        // Static training state = 14 + 14 + 56 = 84 GB
        var batchSizes = ['Batch 1 (Vanilla)', 'Batch 1 (Checkpointed)', 'Batch 4 (Vanilla)', 'Batch 4 (Checkpointed)', 'Batch 8 (Vanilla)', 'Batch 8 (Checkpointed)'];
        var weights = [14, 14, 14, 14, 14, 14];
        var grads = [14, 14, 14, 14, 14, 14];
        var optStates = [56, 56, 56, 56, 56, 56];
        // Activation memory: ~24 bytes/token/layer * L * s * b (vanilla) vs sqrt(L) (checkpointed)
        var activations = [12, 3, 48, 6, 96, 9];

        var data = [
            { x: batchSizes, y: weights, name: 'Weights (BF16, 2 bytes/param)', type: 'bar', marker: { color: '#286b82' } },
            { x: batchSizes, y: grads, name: 'Gradients (BF16, 2 bytes/param)', type: 'bar', marker: { color: '#519db8' } },
            { x: batchSizes, y: optStates, name: 'AdamW States (FP32 moments, 8 bytes/param)', type: 'bar', marker: { color: '#e69f00' } },
            { x: batchSizes, y: activations, name: 'Activations', type: 'bar', marker: { color: '#b84a39' } }
        ];

        var layout = {
            title: { text: '<b>Training Memory Footprint: 7B Model (Standard vs Activation Checkpointing)</b>', x: 0.5, font: { size: 15 } },
            barmode: 'stack',
            xaxis: { title: 'Execution Configuration', gridcolor: '#e5e5e5' },
            yaxis: { title: 'GPU Memory Required (GB)', gridcolor: '#e5e5e5' },
            height: 460,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 70, l: 65, r: 35 },
            legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture2Charts = function () {
        drawRoofline('plotly-cs336-2-roofline');
        drawMemoryBreakdown('plotly-cs336-2-memory-breakdown');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture2Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-2-roofline')) {
        window.renderCs336Lecture2Charts();
    }
})();
