/* ==========================================================================
   CS336 Guest Lecture interactive Plotly charts — Systems & Alternative Architectures (Dan Fu).
   Chart 1: Disaggregated Prefill-Decode Cluster Throughput vs Colocated
   Chart 2: Mega-Kernel Execution Time vs Fragmented Multi-Kernel Launches
   ========================================================================== */

(function () {
    'use strict';

    function drawDisaggregated(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var concurrencies = [16, 32, 64, 128, 256, 512];
        var xLabels = concurrencies.map(function (c) { return c + ' Concur'; });

        // Colocated serving (mixing prefill and decode on same GPU): context switches cause throughput bottleneck
        var colocatedThroughput = [350, 620, 1050, 1600, 2100, 2400]; // tokens/s
        // Disaggregated cluster (dedicated prefill nodes + dedicated decode nodes with KV cache streaming)
        var disaggregatedThroughput = [420, 820, 1580, 2900, 4800, 6800]; // tokens/s

        var data = [
            {
                x: xLabels,
                y: disaggregatedThroughput,
                mode: 'lines+markers',
                name: 'Disaggregated Serving (Dedicated Prefill / Decode Nodes)',
                line: { color: '#286b82', width: 3.5 },
                marker: { size: 8 }
            },
            {
                x: xLabels,
                y: colocatedThroughput,
                mode: 'lines+markers',
                name: 'Colocated Serving (Standard Shared Cluster)',
                line: { color: '#b84a39', width: 2.5, dash: 'dash' },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>Disaggregated Serving: Cluster Generation Throughput vs Concurrency</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Concurrent User Request Streams', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Total Tokens Generated / sec', gridcolor: '#e5e5e5' },
            annotations: [
                {
                    x: '256 Concur',
                    y: 4800,
                    text: '<b>2.3x Throughput Gain</b><br>Eliminates SM starvation during decode',
                    showarrow: true,
                    arrowhead: 2,
                    ax: -70,
                    ay: -35,
                    font: { size: 11, color: '#286b82' }
                }
            ],
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawMegaKernel(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Stage breakdown of forward pass execution (in microseconds)
        var stages = ['QKV Projection', 'Inter-Kernel Launch & SM Drain', 'Attention Computation', 'Output Projection (W_O)', 'LayerNorm & Residual Add'];
        // Standard Multi-Kernel execution has launch gaps and HBM writes between every kernel
        var standardMultiKernel = [45, 28, 85, 45, 22]; // Total = 225 us
        // Mega-Kernel (ThunderKittens) fuses entire layer into single grid with register pipelining
        var megaKernelFused = [40, 0, 65, 38, 5]; // Total = 148 us (~35% lower latency)

        var data = [
            {
                x: stages,
                y: standardMultiKernel,
                name: 'Standard Fragmented PyTorch / CUDA Kernels (225 μs)',
                type: 'bar',
                marker: { color: '#b84a39' }
            },
            {
                x: stages,
                y: megaKernelFused,
                name: 'Mega-Kernel (ThunderKittens Layer Fusion: 148 μs)',
                type: 'bar',
                marker: { color: '#286b82' }
            }
        ];

        var layout = {
            title: { text: '<b>Mega-Kernel Layer Fusion vs Standard Multi-Kernel Execution Time Breakdown</b>', x: 0.5, font: { size: 15 } },
            barmode: 'group',
            xaxis: { gridcolor: '#e5e5e5' },
            yaxis: { title: 'Execution Latency (μs)', gridcolor: '#e5e5e5' },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 70, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336LectureGuestCharts = function () {
        drawDisaggregated('plotly-cs336-guest-disaggregated');
        drawMegaKernel('plotly-cs336-guest-megakernel');
    };

    document.addEventListener('markdown:rendered', window.renderCs336LectureGuestCharts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-guest-disaggregated')) {
        window.renderCs336LectureGuestCharts();
    }
})();
