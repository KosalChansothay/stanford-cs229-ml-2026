/* ==========================================================================
   CS231N Lecture 11 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-11.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Memory per GPU vs sequence length for DP / DP+checkpointing / CP. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        var Ss = [512, 1024, 2048, 4096, 8192, 16384, 32768];
        function mem(strategy, S) {
            var act = S * 0.012, states = 8;
            if (strategy === 'dp') return act + states;
            if (strategy === 'ckpt') return Math.sqrt(S) * 0.05 + states;
            return act / 8 + states;
        }
        var data = [
            { x: Ss, y: Ss.map(function (S) { return mem('dp', S); }), mode: 'lines+markers', name: 'Data parallel', line: { color: '#ff7b72', width: 3 } },
            { x: Ss, y: Ss.map(function (S) { return mem('ckpt', S); }), mode: 'lines+markers', name: 'DP + activation checkpointing', line: { color: '#79c0ff', width: 3 } },
            { x: Ss, y: Ss.map(function (S) { return mem('cp', S); }), mode: 'lines+markers', name: 'Context parallel (8-way)', line: { color: '#7ee787', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>GPU Memory vs Sequence Length by Parallelism Strategy</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { type: 'log', title: 'Sequence length (log)', tickmode: 'array', tickvals: Ss, ticktext: Ss.map(function (s) { return s >= 1024 ? (s / 1024) + 'K' : String(s);, gridcolor: '#e5e5e5' }) },
            yaxis: { type: 'log', title: 'Memory (GB, log)', tickmode: 'array', tickvals: [8, 16, 32, 64, 128, 256, 512], ticktext: ['8', '16', '32', '64', '128', '256', '512'], gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture11Charts = function () {
        drawBlueprint('plotly-cs231n-11-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture11Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-11-blueprint')) {
        window.renderCs231nLecture11Charts();
    }
})();
