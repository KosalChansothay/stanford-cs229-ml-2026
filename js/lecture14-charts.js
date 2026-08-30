/* ==========================================================================
   Lecture 14 interactive Plotly charts — Transformers & Attention.
   Chart 1: Causal attention heatmap — softmax(QK^T/sqrt(d_h) + M) with the
            lower-triangular causal mask, computed for real random Q,K.
   Chart 2: Attention compute/memory O(T^2) vs FlashAttention HBM traffic
            O(T) — the hardware bottleneck.
   Loaded by courses/cs229/lecture-14.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in lecture-14.html).
   ========================================================================== */

(function () {
    'use strict';

    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
            var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    function gaussian(rand) {
        var u = Math.max(rand(), 1e-12), v = rand();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    /* ======================================================================
       Chart 1: Causal attention probability heatmap.
       H^out = softmax(QK^T / sqrt(d_h) + M) V — we visualize the P matrix.
       ====================================================================== */
    function drawCausalAttention(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = mulberry32(14);
        var T = 16, dModel = 32, dHead = 16;

        /* Random input H (T x dModel) and projections. */
        function mat(rows, cols) {
            var m = [];
            for (var i = 0; i < rows; i++) {
                var r = [];
                for (var j = 0; j < cols; j++) r.push(gaussian(rand));
                m.push(r);
            }
            return m;
        }
        function matmul(A, B) {
            var n = A.length, k = B.length, m = B[0].length, out = [];
            for (var i = 0; i < n; i++) {
                var row = [];
                for (var j = 0; j < m; j++) {
                    var s = 0;
                    for (var l = 0; l < k; l++) s += A[i][l] * B[l][j];
                    row.push(s);
                }
                out.push(row);
            }
            return out;
        }
        function randMat(rows, cols, scale) {
            var m = mat(rows, cols);
            return m.map(function (r) { return r.map(function (v) { return v * scale; }); });
        }

        var H = mat(T, dModel);
        var WQ = randMat(dModel, dHead, 0.25);
        var WK = randMat(dModel, dHead, 0.25);
        var Q = matmul(H, WQ), K = matmul(H, WK);

        /* Raw scores + causal mask + softmax per row. */
        var P = [];
        for (var t = 0; t < T; t++) {
            var row = [];
            var maxScore = -Infinity;
            for (var j = 0; j < T; j++) {
                var s = j <= t ? Q[t].reduce(function (acc, v, l) { return acc + v * K[j][l]; }, 0) / Math.sqrt(dHead) : -Infinity;
                row.push(s);
                if (s > maxScore) maxScore = s;
            }
            var Z = 0;
            row = row.map(function (s) { var e = s === -Infinity ? 0 : Math.exp(s - maxScore); Z += e; return e; });
            P.push(row.map(function (e) { return e / Z; }));
        }

        var data = [{
            z: P, x: P.map(function (_, i) { return 'k' + i; }), y: P.map(function (_, i) { return 'q' + i; }),
            type: 'heatmap', colorscale: 'Blues',
            hovertemplate: 'query %{y} attends to key %{x}: %{z:.3f}<extra></extra>',
            colorbar: { title: { text: 'P' }, thickness: 14 }
        }];

        var layout = {
            title: { text: '<b>Causal Attention Matrix P = softmax(QKᵀ/√d_h + M)</b>', x: 0.5, font: { size: 16 } },
            xaxis: { title: 'Keys (past & present only)', tickfont: { size: 8 } },
            yaxis: { title: 'Queries', tickfont: { size: 8 }, autorange: 'reversed' },
            height: 560,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 55 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: O(T^2) attention matrix HBM traffic vs O(T) FlashAttention.
       Log-scale plot of memory moved (in units of T^2 * 2 bytes vs ~T*d).
       ====================================================================== */
    function drawAttentionScaling(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var Ts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
        var labels = Ts.map(function (t) { return t >= 1024 ? (t / 1024) + 'k' : t; });
        var quad = Ts.map(function (t) { return t * t; });
        var lin = Ts.map(function (t) { return t * 128; }); /* FlashAttention HBM traffic ~ O(T·d) */

        var data = [
            {
                x: labels, y: quad, type: 'scatter', mode: 'lines+markers',
                name: 'Standard attention: T² HBM reads/writes',
                line: { color: '#c0392b', width: 3 },
                hovertemplate: 'T=%{x}: %{y:.3s} units<extra>standard</extra>'
            },
            {
                x: labels, y: lin, type: 'scatter', mode: 'lines+markers',
                name: 'FlashAttention: O(T) HBM traffic (tiled in SRAM)',
                line: { color: '#2e7d4f', width: 3 },
                hovertemplate: 'T=%{x}: %{y:.3s} units<extra>FlashAttention</extra>'
            }
        ];

        var layout = {
            title: { text: '<b>Attention Memory Bottleneck: O(T²) vs O(T) HBM Traffic</b>', x: 0.5, font: { size: 16 } },
            xaxis: { title: 'Sequence length T', gridcolor: 'lightgray' },
            yaxis: { title: 'HBM memory traffic (log scale)', type: 'log', gridcolor: 'lightgray' },
            height: 460,
            responsive: true,
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.75)', bordercolor: 'lightgray', borderwidth: 1, font: { size: 11 } },
            plot_bgcolor: 'white',
            margin: { t: 55 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderLecture14Charts = function () {
        drawCausalAttention('plotly-14-causal-attention');
        drawAttentionScaling('plotly-14-attention-scaling');
    };

    document.addEventListener('markdown:rendered', window.renderLecture14Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-14-causal-attention')) {
        window.renderLecture14Charts();
    }
})();
