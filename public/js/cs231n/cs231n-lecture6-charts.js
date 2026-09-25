/* ==========================================================================
   CS231N Lecture 6 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-6.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Activation statistics vs depth: unnormalized vs BatchNorm. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(11);
        var depth = 25;
        function stats(gain, norm) {
            var means = [], stds = [], v = 1;
            for (var L = 0; L < depth; L++) {
                if (!norm) v = v * gain;
                means.push(norm ? 0 : v * 0.1);
                stds.push(norm ? 1 : Math.abs(v));
            }
            return { means: means, stds: stds };
        }
        var noBN = stats(1.4, false), withBN = stats(1.4, true);
        var Ls = [], L;
        for (L = 0; L < depth; L++) Ls.push(L);
        var data = [
            { x: Ls, y: noBN.stds, mode: 'lines+markers', name: 'No norm: activation std', line: { color: '#ff7b72', width: 3 } },
            { x: Ls, y: withBN.stds, mode: 'lines+markers', name: 'BatchNorm: activation std', line: { color: '#7ee787', width: 3 } },
            { x: Ls, y: noBN.means, mode: 'lines', name: 'No norm: mean (small)', line: { color: '#f0883e', width: 2, dash: 'dot' } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>BatchNorm Stabilizes Activation Scale Across Depth</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Layer index', gridcolor: '#e5e5e5' },
            yaxis: { type: 'log', title: 'Activation std (log scale)', tickmode: 'array', tickvals: [1, 100, 1e4, 1e6, 1e8, 1e10], ticktext: ['1', '10^2', '10^4', '10^6', '10^8', '10^10'], gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture6Charts = function () {
        drawBlueprint('plotly-cs231n-6-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture6Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-6-blueprint')) {
        window.renderCs231nLecture6Charts();
    }
})();
