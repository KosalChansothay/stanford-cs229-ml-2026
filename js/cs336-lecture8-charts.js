/* ==========================================================================
   CS336 Lecture 8 interactive Plotly charts — Parallelism (Advanced).
   Chart 1: ZeRO Stages (ZeRO-0, 1, 2, 3 / FSDP) Memory Footprint per GPU
   Chart 2: Pipeline Parallelism (1F1B) Bubble Overhead vs Micro-batches m
   ========================================================================== */

(function () {
    'use strict';

    function drawZeroMemory(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Model: 70B parameter dense model (N = 70e9). Cluster: 64 GPUs.
        // In FP16/BF16:
        // Weights = 2N = 140 GB, Gradients = 2N = 140 GB, Optimizer = 12N = 840 GB. Total static = 1120 GB.
        // ZeRO-0 (Standard DDP): Each GPU replicates all weights, grads, opt states = 1120 GB (OOM on 80GB GPU)
        // ZeRO-1 (Shard Optimizer across 64 GPUs): Weights 140 GB, Grads 140 GB, Opt states 840/64 = 13.1 GB -> Total 293.1 GB
        // ZeRO-2 (Shard Opt + Grads across 64 GPUs): Weights 140 GB, Grads 140/64 = 2.2 GB, Opt 13.1 GB -> Total 155.3 GB
        // ZeRO-3 / FSDP (Shard Weights + Grads + Opt across 64 GPUs): Weights 140/64 = 2.2 GB, Grads 2.2 GB, Opt 13.1 GB -> Total 17.5 GB
        var stages = ['ZeRO-0 (Standard DDP)', 'ZeRO-1 (Shard Optim)', 'ZeRO-2 (Shard Optim + Grad)', 'ZeRO-3 / FSDP (Full Shard)'];
        var optMemory = [840, 840 / 64, 840 / 64, 840 / 64];
        var gradMemory = [140, 140, 140 / 64, 140 / 64];
        var weightMemory = [140, 140, 140, 140 / 64];

        var data = [
            {
                x: stages,
                y: weightMemory,
                name: 'Model Weights (140 GB total)',
                type: 'bar',
                marker: { color: '#286b82' }
            },
            {
                x: stages,
                y: gradMemory,
                name: 'Gradients (140 GB total)',
                type: 'bar',
                marker: { color: '#519db8' }
            },
            {
                x: stages,
                y: optMemory,
                name: 'AdamW States (840 GB total)',
                type: 'bar',
                marker: { color: '#e69f00' }
            }
        ];

        var layout = {
            title: { text: '<b>Memory per GPU (70B Model on 64 × H100 GPUs) across ZeRO Stages</b>', x: 0.5, font: { size: 15 } },
            barmode: 'stack',
            xaxis: { gridcolor: '#e5e5e5' },
            yaxis: { title: 'Static Training Memory per GPU (GB)', gridcolor: '#e5e5e5', type: 'log' },
            height: 460,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 60, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawPipelineBubble(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Bubble fraction F_bubble = (p - 1) / m for 1F1B pipeline schedule
        // where p = pipeline stages (depth), m = number of micro-batches
        var microBatches = [4, 8, 16, 32, 64, 128];
        var xLabels = microBatches.map(function (m) { return 'm=' + m; });

        var depths = [4, 8, 16];
        var colors = ['#3d8b57', '#286b82', '#b84a39'];

        var data = depths.map(function (p, idx) {
            var bubbleFractions = microBatches.map(function (m) {
                return ((p - 1) / m) * 100; // Percentage
            });
            return {
                x: xLabels,
                y: bubbleFractions,
                mode: 'lines+markers',
                name: 'Pipeline Depth p = ' + p,
                line: { color: colors[idx], width: 3 },
                marker: { size: 7 }
            };
        });

        var layout = {
            title: { text: '<b>Pipeline Parallelism (1F1B) Bubble Overhead: F_bubble = (p - 1) / m</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Number of Micro-batches (m)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Idle Bubble Overhead (%)', gridcolor: '#e5e5e5', range: [0, 80] },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture8Charts = function () {
        drawZeroMemory('plotly-cs336-8-zero-memory');
        drawPipelineBubble('plotly-cs336-8-pipeline-bubble');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture8Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-8-zero-memory')) {
        window.renderCs336Lecture8Charts();
    }
})();
