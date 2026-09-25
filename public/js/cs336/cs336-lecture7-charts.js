/* ==========================================================================
   CS336 Lecture 7 interactive Plotly charts — Parallelism (Foundations).
   Chart 1: Ring AllReduce Communication Volume vs Cluster Size P
   Chart 2: Tensor Parallelism Speedup: NVLink (900 GB/s) vs PCIe (64 GB/s)
   ========================================================================== */

(function () {
    'use strict';

    function drawRingAllreduce(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var numRanks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        // Data volume transferred per GPU in units of message size M
        // Ring AllReduce: 2 * (P - 1) / P * M -> approaches 2M asymptotically
        var ringVolume = numRanks.map(function (P) { return 2 * (P - 1) / P; });
        // Naive AllReduce (all-to-all centralized): 2 * (P - 1) * M
        var naiveVolume = numRanks.map(function (P) { return 2 * (P - 1); });

        var data = [
            {
                x: numRanks,
                y: ringVolume,
                mode: 'lines+markers',
                name: 'Ring AllReduce (Bandwidth Optimal: ~2M)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            },
            {
                x: numRanks.slice(0, 5),
                y: naiveVolume.slice(0, 5),
                mode: 'lines+markers',
                name: 'Naive AllReduce O(P · M)',
                line: { color: '#b84a39', width: 2.5, dash: 'dash' },
                marker: { size: 6 }
            }
        ];

        var layout = {
            title: { text: '<b>Ring AllReduce: Data Sent per GPU vs Cluster Size P</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: {
                title: 'Number of GPUs (P)',
                gridcolor: '#e5e5e5',
                type: 'log',
                tickmode: 'array',
                tickvals: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                ticktext: ['2', '4', '8', '16', '32', '64', '128', '256', '512', '1,024']
            },
            yaxis: { title: 'Bytes Sent per GPU (Multiples of Message Size M)', gridcolor: '#e5e5e5' },
            annotations: [
                {
                    x: Math.log10(128),
                    y: 2.0,
                    text: '<b>Asymptotic Ceiling: 2M</b><br>Communication cost is constant as P grows',
                    showarrow: true,
                    arrowhead: 2,
                    ax: -70,
                    ay: -40,
                    font: { size: 11, color: '#286b82' }
                }
            ],
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawTensorParallel(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var tpDegrees = [1, 2, 4, 8];
        var idealSpeedup = [1, 2, 4, 8];
        // NVLink (900 GB/s intra-node): high efficiency (~7.2x on 8 GPUs)
        var nvlinkSpeedup = [1, 1.95, 3.8, 7.2];
        // PCIe / Cross-Node (64 GB/s): communication bottleneck degrades speedup (~2.4x on 8 GPUs)
        var pcieSpeedup = [1, 1.45, 1.9, 2.3];

        var data = [
            {
                x: tpDegrees.map(function (d) { return 'TP=' + d; }),
                y: idealSpeedup,
                mode: 'lines',
                name: 'Ideal Linear Speedup',
                line: { color: '#999', width: 2, dash: 'dash' }
            },
            {
                x: tpDegrees.map(function (d) { return 'TP=' + d; }),
                y: nvlinkSpeedup,
                mode: 'lines+markers',
                name: 'Intra-Node NVLink (900 GB/s)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 8 }
            },
            {
                x: tpDegrees.map(function (d) { return 'TP=' + d; }),
                y: pcieSpeedup,
                mode: 'lines+markers',
                name: 'Inter-Node / PCIe (64 GB/s)',
                line: { color: '#b84a39', width: 3 },
                marker: { size: 8 }
            }
        ];

        var layout = {
            title: { text: '<b>Megatron Tensor Parallelism Speedup: NVLink vs PCIe Bottleneck</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Tensor Parallel Degree', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Throughput Speedup Factor', gridcolor: '#e5e5e5', range: [0, 8.5] },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture7Charts = function () {
        drawRingAllreduce('plotly-cs336-7-ring-allreduce');
        drawTensorParallel('plotly-cs336-7-tensor-parallel');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture7Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-7-ring-allreduce')) {
        window.renderCs336Lecture7Charts();
    }
})();
