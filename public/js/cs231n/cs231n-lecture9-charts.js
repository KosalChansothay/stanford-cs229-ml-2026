/* ==========================================================================
   CS231N Lecture 9 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-9.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Max-unpooling vs nearest-neighbor upsampling reconstruction error. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(5);
        var factor = [1, 2, 4, 8, 16, 32];
        var nn = factor.map(function (f) { return 0.9 - 0.25 * Math.log2(f) + 0.05 * rnd(); });
        var mu = factor.map(function (f) { return 0.75 - 0.3 * Math.log2(f) + 0.03 * rnd(); });
        var tconv = factor.map(function (f) { return 0.65 - 0.28 * Math.log2(f) + 0.02 * rnd(); });
        var data = [
            { x: factor, y: nn, mode: 'lines+markers', name: 'Nearest neighbor', line: { color: '#8b949e', width: 2.5 } },
            { x: factor, y: mu, mode: 'lines+markers', name: 'Max unpooling', line: { color: '#79c0ff', width: 3 } },
            { x: factor, y: tconv, mode: 'lines+markers', name: 'Transposed conv', line: { color: '#7ee787', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Segmentation Upsampling Quality vs Output Resolution</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { type: 'log', title: 'Upsampling factor (log)', tickmode: 'array', tickvals: factor, ticktext: factor.map(String), gridcolor: '#e5e5e5' },
            yaxis: { title: 'Boundary F-score (synthetic)', gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture9Charts = function () {
        drawBlueprint('plotly-cs231n-9-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture9Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-9-blueprint')) {
        window.renderCs231nLecture9Charts();
    }
})();
