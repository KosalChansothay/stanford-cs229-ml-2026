/* ==========================================================================
   CS231N Lecture 17 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-17.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Behavior cloning drift vs DAgger self-correction over rollout steps. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(13);
        var T = 30, ts = [], i;
        for (i = 0; i < T; i++) ts.push(i);
        var bc = [0], dag = [0];
        var bcV = 0, dagV = 0;
        for (i = 1; i < T; i++) {
            bcV += 0.35 + 0.2 * rnd(); bc.push(bcV);
            dagV = Math.max(0, dagV + (0.18 - 0.012 * i) + 0.15 * (rnd() - 0.5)); dag.push(dagV);
        }
        var data = [
            { x: ts, y: bc, mode: 'lines+markers', name: 'Behavior cloning: distance from expert', line: { color: '#ff7b72', width: 3 } },
            { x: ts, y: dag, mode: 'lines+markers', name: 'DAgger: distance from expert', line: { color: '#7ee787', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Covariate Shift: BC Drifts Out-of-Distribution, DAgger Recovers</b>', x: 0.5, y: 0.98, font: { size: 14 } },
            xaxis: { title: 'Rollout step', gridcolor: '#e5e5e5' }, yaxis: { title: 'State distance from expert trajectory', gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture17Charts = function () {
        drawBlueprint('plotly-cs231n-17-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture17Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-17-blueprint')) {
        window.renderCs231nLecture17Charts();
    }
})();
