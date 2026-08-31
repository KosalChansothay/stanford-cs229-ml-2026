/* ==========================================================================
   CS231N Lecture 8 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-8.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Attention weight heatmap + quadratic cost growth with sequence length. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(31);
        var N = 12, tokens = [], i, j;
        for (i = 0; i < N; i++) tokens.push('t' + i);
        var z = [];
        for (i = 0; i < N; i++) {
            var row = [], s = 0;
            for (j = 0; j < N; j++) { var w = Math.exp(-0.35 * Math.abs(i - j)) + 0.15 * rnd(); row.push(w); s += w; }
            for (j = 0; j < N; j++) row[j] /= s;
            z.push(row);
        }
        var Ls = [10, 20, 50, 100, 200, 500, 1000, 2000];
        var cost = Ls.map(function (L) { return L * L; });
        var data = [
            { type: 'heatmap', x: tokens, y: tokens.slice().reverse(), z: z, colorscale: 'Viridis', hovertemplate: 'query %{y} attends %{x}<br>w %{z:.3f}<extra></extra>', colorbar: { title: { text: 'weight' } } },
            { x: Ls, y: cost, mode: 'lines+markers', name: 'O(N^2) attention cost', visible: false, line: { color: '#ff7b72', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Self-Attention Weights (local + content mix)</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Key token', gridcolor: '#e5e5e5' }, yaxis: { title: 'Query token', gridcolor: '#e5e5e5' }, height: 520, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
        layout.updatemenus = [{ type: 'buttons', direction: 'right', x: 0.6, y: 1.14, buttons: [
            { label: 'Heatmap', method: 'restyle', args: [{ visible: [true, false] }] },
            { label: 'Cost O(N^2)', method: 'restyle', args: [{ visible: [false, true] }] }
        ] }];
        layout.yaxis2 = { title: 'Pairwise ops', overlaying: 'y', side: 'right', visible: false };
        Plotly.react(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture8Charts = function () {
        drawBlueprint('plotly-cs231n-8-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture8Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-8-blueprint')) {
        window.renderCs231nLecture8Charts();
    }
})();
