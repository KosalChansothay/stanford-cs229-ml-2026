/* ==========================================================================
   CS231N Lecture 5 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-5.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Conv output shape / params / FLOPs as a function of K, P, S. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(7);
        var H = 32, Cin = 8, Cout = 16;
        var Ks = [1, 3, 5, 7, 9, 11];
        function calc(K, P, S) {
            var out = Math.floor((H + 2 * P - K) / S) + 1;
            var params = K * K * Cin * Cout + Cout;
            var flops = 2 * out * out * K * K * Cin * Cout;
            return { out: out, params: params, flops: flops };
        }
        var traces = [], combos = [];
        [0, 1, 2].forEach(function (S) {
            var xs = [], ys = [], cd = [];
            Ks.forEach(function (K) {
                var P = Math.floor(K / 2);
                var r = calc(K, P, S + 1);
                xs.push('K=' + K); ys.push(r.flops / 1e6); cd.push(r.out);
            });
            traces.push({ x: xs, y: ys, type: 'bar', name: 'Stride ' + (S + 1), customdata: cd, hovertemplate: '%{x}, stride ' + (S + 1) + '<br>MFLOPs %{y:.1f}<br>out %{customdata}x%{customdata}<extra></extra>' });
        });
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Convolution Cost: MFLOPs vs Kernel Size and Stride (H=32, Cin=8, Cout=16)</b>', x: 0.5, y: 0.98, font: { size: 14 } },
            xaxis: { title: 'Kernel size (same padding)', gridcolor: '#e5e5e5' }, yaxis: { title: 'MFLOPs per forward pass', gridcolor: '#e5e5e5' },
            barmode: 'group', height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, traces, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture5Charts = function () {
        drawBlueprint('plotly-cs231n-5-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture5Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-5-blueprint')) {
        window.renderCs231nLecture5Charts();
    }
})();
