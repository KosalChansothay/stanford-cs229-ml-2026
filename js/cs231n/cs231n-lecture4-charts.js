/* ==========================================================================
   CS231N Lecture 4 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-4.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Vanishing gradients: per-layer gradient magnitude through sigmoid vs ReLU. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function grads(activation) {
            var depth = 20, g = [1], v = 1;
            for (var L = 1; L < depth; L++) {
                var local = activation === 'sigmoid' ? 0.25 : 1.0;
                v = v * local; g.push(v);
            }
            return g;
        }
        var Ls = [], d;
        for (d = 0; d < 20; d++) Ls.push(d);
        var data = [
            { x: Ls, y: grads('sigmoid'), mode: 'lines+markers', name: 'Sigmoid (max local grad 0.25)', line: { color: '#ff7b72', width: 3 } },
            { x: Ls, y: grads('relu'), mode: 'lines+markers', name: 'ReLU (local grad 1)', line: { color: '#7ee787', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Vanishing Gradients: Magnitude vs Depth</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Layer depth (backprop steps from loss)', gridcolor: '#e5e5e5' },
            yaxis: { type: 'log', title: 'Gradient magnitude (log scale)', tickmode: 'array', tickvals: [1e-12, 1e-9, 1e-6, 1e-3, 1], ticktext: ['10^-12', '10^-9', '10^-6', '10^-3', '1'], gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture4Charts = function () {
        drawBlueprint('plotly-cs231n-4-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture4Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-4-blueprint')) {
        window.renderCs231nLecture4Charts();
    }
})();
