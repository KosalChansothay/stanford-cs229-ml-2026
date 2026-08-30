/* ==========================================================================
   CS336 Lecture 3 interactive Plotly charts — Architectures.
   Chart 1: Grouped-Query Attention (GQA) KV Cache Memory Compression
   Chart 2: Rotary Position Embedding (RoPE) Relative Rotation Dynamics
   ========================================================================== */

(function () {
    'use strict';

    function drawGqaCompression(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Model config: LLaMA-style 70B (80 layers, d_head = 128, H_q = 64)
        // Memory per token for KV cache (bytes): 2 * 2 * n_layers * n_kv_heads * d_head
        // In FP16/BF16 (2 bytes per entry, factor 2 for K and V)
        var contextLengths = [4096, 8192, 16384, 32768, 65536, 131072];
        var xLabels = ['4k', '8k', '16k', '32k', '64k', '128k'];

        // MHA: H_kv = 64
        var mhaBytesPerToken = 2 * 2 * 80 * 64 * 128; // 2,621,440 bytes/tok = 2.5 MB/tok
        // GQA: H_kv = 8 (8x compression)
        var gqaBytesPerToken = 2 * 2 * 80 * 8 * 128; // 327,680 bytes/tok = 0.3125 MB/tok
        // MQA: H_kv = 1 (64x compression)
        var mqaBytesPerToken = 2 * 2 * 80 * 1 * 128; // 40,960 bytes/tok = 0.039 MB/tok

        var mhaGb = contextLengths.map(function (c) { return (c * mhaBytesPerToken) / 1e9; });
        var gqaGb = contextLengths.map(function (c) { return (c * gqaBytesPerToken) / 1e9; });
        var mqaGb = contextLengths.map(function (c) { return (c * mqaBytesPerToken) / 1e9; });

        var data = [
            {
                x: xLabels,
                y: mhaGb,
                mode: 'lines+markers',
                name: 'MHA (64 KV Heads)',
                line: { color: '#b84a39', width: 3 },
                marker: { size: 7 }
            },
            {
                x: xLabels,
                y: gqaGb,
                mode: 'lines+markers',
                name: 'GQA (8 KV Heads — LLaMA-3 Standard)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            },
            {
                x: xLabels,
                y: mqaGb,
                mode: 'lines+markers',
                name: 'MQA (1 KV Head)',
                line: { color: '#3d8b57', width: 3 },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>KV Cache Footprint vs Context Length (70B Model, 1 Sequence)</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Context Window (Tokens)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'KV Cache Size (GB)', gridcolor: '#e5e5e5' },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawRopeInvariance(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Simulate RoPE inner product <R_m q, R_n k> as a function of relative distance delta = m - n
        // For standard base theta = 10000, 2D rotation frequency omega = theta^(-2i/d)
        var distances = [];
        for (var d = 0; d <= 64; d++) distances.push(d);

        // Compute simulated attention score decay over relative distance across multiple head dimensions
        var theta = 10000;
        var scores = distances.map(function (delta) {
            // Sum cosine across representative 2D frequency bands
            var sum = 0;
            var numBands = 8;
            for (var i = 0; i < numBands; i++) {
                var omega = Math.pow(theta, -2 * i / 64);
                sum += Math.cos(delta * omega);
            }
            return sum / numBands;
        });

        var data = [{
            x: distances,
            y: scores,
            mode: 'lines',
            name: 'Expected Attention Weight <R_m q, R_n k>',
            line: { color: '#286b82', width: 3 },
            fill: 'tozeroy',
            fillcolor: 'rgba(40, 107, 130, 0.15)'
        }];

        var layout = {
            title: { text: '<b>RoPE Relative Attention Score vs Token Distance (m - n)</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Relative Token Distance |m - n|', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Relative Inner Product Score', gridcolor: '#e5e5e5', range: [-0.3, 1.05] },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture3Charts = function () {
        drawGqaCompression('plotly-cs336-3-gqa-compression');
        drawRopeInvariance('plotly-cs336-3-rope-invariance');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture3Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-3-gqa-compression')) {
        window.renderCs336Lecture3Charts();
    }
})();
