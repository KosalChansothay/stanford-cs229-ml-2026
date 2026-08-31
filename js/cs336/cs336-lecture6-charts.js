/* ==========================================================================
   CS336 Lecture 6 interactive Plotly charts — Kernels, Triton, XLA.
   Chart 1: Matrix Multiplication Arithmetic Intensity vs Dimension N
   Chart 2: Triton Fused Operator Speedup & Memory Traffic vs PyTorch Naive
   ========================================================================== */

(function () {
    'use strict';

    function drawMatmulIntensity(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var dimensions = [64, 128, 256, 512, 895, 1024, 2048, 4096, 8192];
        var intensities = dimensions.map(function (N) { return N / 3; });
        var threshold = 295.37; // H100 BF16 intensity ceiling

        var data = [
            {
                x: dimensions,
                y: intensities,
                mode: 'lines+markers',
                name: 'MatMul Arithmetic Intensity (N/3)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            },
            {
                x: [64, 8192],
                y: [threshold, threshold],
                mode: 'lines',
                name: 'H100 Hardware Ceiling (295.4 FLOPs/B)',
                line: { color: '#b84a39', width: 2.5, dash: 'dash' }
            }
        ];

        var layout = {
            title: { text: '<b>MatMul Arithmetic Intensity (N/3) & H100 Compute Boundary</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: {
                title: 'Matrix Dimension N (Square GEMM N × N)',
                type: 'log',
                tickmode: 'array',
                tickvals: [64, 128, 256, 512, 1024, 2048, 4096, 8192],
                ticktext: ['64', '128', '256', '512', '1,024', '2,048', '4,096', '8,192'],
                gridcolor: '#e5e5e5'
            },
            yaxis: {
                title: 'Arithmetic Intensity (FLOPs / Byte)',
                type: 'log',
                tickmode: 'array',
                tickvals: [20, 50, 100, 200, 295.4, 500, 1000, 2000, 3000],
                ticktext: ['20', '50', '100', '200', '295.4', '500', '1,000', '2,000', '3,000'],
                gridcolor: '#e5e5e5'
            },
            annotations: [
                {
                    x: Math.log10(895),
                    y: Math.log10(295.4),
                    text: '<b>Boundary: N ≈ 896</b><br>Transitions to Compute-Bound',
                    showarrow: true,
                    arrowhead: 2,
                    ax: -60,
                    ay: -40,
                    font: { size: 11, color: '#b84a39' }
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

    function drawTritonFusion(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var seqLengths = ['1k', '2k', '4k', '8k', '16k'];
        // Execution time in microseconds for Fused Softmax / LayerNorm
        var naiveTime = [42, 85, 175, 360, 740];
        var tritonFusedTime = [11, 21, 43, 87, 178];

        var data = [
            {
                x: seqLengths,
                y: naiveTime,
                name: 'Unfused PyTorch (Multiple Kernel Launches + HBM RTT)',
                type: 'bar',
                marker: { color: '#b84a39' }
            },
            {
                x: seqLengths,
                y: tritonFusedTime,
                name: 'Triton Fused Kernel (SRAM Block Accumulation, 4x Speedup)',
                type: 'bar',
                marker: { color: '#286b82' }
            }
        ];

        var layout = {
            title: { text: '<b>Triton Operator Fusion Execution Latency vs Sequence Length</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            barmode: 'group',
            xaxis: { title: 'Batch Sequence Length', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Kernel Execution Latency (μs)', gridcolor: '#e5e5e5' },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture6Charts = function () {
        drawMatmulIntensity('plotly-cs336-6-matmul-intensity');
        drawTritonFusion('plotly-cs336-6-triton-fusion');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture6Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-6-matmul-intensity')) {
        window.renderCs336Lecture6Charts();
    }
})();
