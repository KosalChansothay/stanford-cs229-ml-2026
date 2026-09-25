/* ==========================================================================
   CS231N Lecture 7 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-7.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Gradient magnitude through time: vanilla RNN vs LSTM additive highway. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        var T = 40, ts = [], i;
        for (i = 0; i < T; i++) ts.push(i);
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(23);
        var rnn = [1], lstm = [1];
        var rnnV = 1, lstmV = 1;
        for (i = 1; i < T; i++) {
            rnnV *= 0.72; rnn.push(rnnV);
            lstmV *= 0.97 + 0.02 * rnd(); lstm.push(Math.min(1.2, lstmV));
        }
        var data = [
            { x: ts, y: rnn, mode: 'lines+markers', name: 'Vanilla RNN gradient', line: { color: '#ff7b72', width: 3 } },
            { x: ts, y: lstm, mode: 'lines+markers', name: 'LSTM gradient highway', line: { color: '#7ee787', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Gradient Flow Through Time: Vanilla RNN vs LSTM</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Time steps back (t)', gridcolor: '#e5e5e5' },
            yaxis: { type: 'log', title: 'Gradient magnitude (log scale)', tickmode: 'array', tickvals: [1e-12, 1e-9, 1e-6, 1e-3, 1], ticktext: ['10^-12', '10^-9', '10^-6', '10^-3', '1'], gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture7Charts = function () {
        drawBlueprint('plotly-cs231n-7-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture7Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-7-blueprint')) {
        window.renderCs231nLecture7Charts();
    }
})();
