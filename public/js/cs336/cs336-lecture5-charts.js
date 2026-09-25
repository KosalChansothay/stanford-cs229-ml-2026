/* ==========================================================================
   CS336 Lecture 5 interactive Plotly charts — GPUs, TPUs, and FlashAttention.
   Chart 1: FlashAttention SRAM Tiling vs Standard Attention HBM Memory Accesses
   Chart 2: Quantization Precision vs Tensor Core Compute Throughput
   ========================================================================== */

(function () {
    'use strict';

    function drawFlashIo(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var seqLengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536];
        var xLabels = ['1k', '2k', '4k', '8k', '16k', '32k', '64k'];
        var d = 128; // head dim
        var M_sram = 100 * 1024; // 100 KB SRAM per block

        // Standard attention HBM memory access: O(N * d + N^2)
        // Writes intermediate S and P matrices to HBM and reads them back
        var standardHbmGb = seqLengths.map(function (N) {
            var bytes = (4 * N * d + 2 * N * N) * 2; // FP16
            return bytes / 1e9;
        });

        // FlashAttention HBM memory access: O(N^2 * d^2 / M_sram) -> scales with N * d
        var flashHbmGb = seqLengths.map(function (N) {
            var bytes = (4 * N * d + (N * N * d * d) / (M_sram * 8)) * 2;
            return bytes / 1e9;
        });

        var data = [
            {
                x: xLabels,
                y: standardHbmGb,
                mode: 'lines+markers',
                name: 'Standard Attention HBM Reads/Writes O(N²)',
                line: { color: '#b84a39', width: 3 },
                marker: { size: 7 }
            },
            {
                x: xLabels,
                y: flashHbmGb,
                mode: 'lines+markers',
                name: 'FlashAttention Tiled HBM Reads/Writes O(N)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>HBM Memory Traffic: FlashAttention (SRAM Tiling) vs Standard Attention</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Sequence Length N (Tokens)', gridcolor: '#e5e5e5' },
            yaxis: {
                title: 'Total HBM Data Transferred (GB)',
                gridcolor: '#e5e5e5',
                type: 'log',
                tickmode: 'array',
                tickvals: [0.01, 0.1, 1, 10, 100, 1000, 10000],
                ticktext: ['10 MB', '100 MB', '1 GB', '10 GB', '100 GB', '1 TB', '10 TB']
            },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawQuantThroughput(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // NVIDIA H100 Tensor Core Peak Speeds (TFLOPs/s) vs Weight Memory Footprint for 70B model (GB)
        var formats = ['FP32 (4 bytes)', 'BF16 / FP16 (2 bytes)', 'FP8 (E4M3/E5M2, 1 byte)', 'FP4 (NV Blackwell, 0.5 byte)'];
        var throughput = [67, 989.5, 1979, 3958]; // Peak TFLOPs/s
        var memoryGb = [280, 140, 70, 35]; // 70B parameters in GB

        var data = [
            {
                x: formats,
                y: throughput,
                name: 'Tensor Core Compute (TFLOPs/s)',
                type: 'bar',
                yaxis: 'y1',
                marker: { color: '#286b82' }
            },
            {
                x: formats,
                y: memoryGb,
                name: '70B Weights Memory Footprint (GB)',
                type: 'bar',
                yaxis: 'y2',
                marker: { color: '#e69f00' }
            }
        ];

        var layout = {
            title: { text: '<b>Quantization Scaling: Peak Tensor Core Compute vs Model Memory Footprint</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { gridcolor: '#e5e5e5' },
            yaxis: { title: 'Peak Compute (TFLOPs/s)', titlefont: { color: '#286b82' }, tickfont: { color: '#286b82' }, gridcolor: '#e5e5e5' },
            yaxis2: { title: '70B Model Weights (GB)', titlefont: { color: '#e69f00' }, tickfont: { color: '#e69f00' }, overlaying: 'y', side: 'right' },
            barmode: 'group',
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 75, r: 105 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture5Charts = function () {
        drawFlashIo('plotly-cs336-5-flash-io');
        drawQuantThroughput('plotly-cs336-5-quant-throughput');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture5Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-5-flash-io')) {
        window.renderCs336Lecture5Charts();
    }
})();
