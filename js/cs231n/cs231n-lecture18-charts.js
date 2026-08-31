/* ==========================================================================
   CS231N Lecture 18 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-18.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Sim vs real success rates across household task categories. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        var tasks = ['Pick', 'Place', 'Pour', 'Fold', 'Open door', 'Wipe', 'Insert', 'Stack'];
        var sim = [0.92, 0.88, 0.74, 0.55, 0.83, 0.68, 0.48, 0.71];
        var real = [0.71, 0.62, 0.45, 0.22, 0.58, 0.4, 0.19, 0.37];
        var data = [
            { x: tasks, y: sim.map(function (v) { return v * 100; }), type: 'bar', name: 'Simulator (privileged)', marker: { color: '#79c0ff' }, hovertemplate: '%{x}: %{y:.0f}%<extra></extra>' },
            { x: tasks, y: real.map(function (v) { return v * 100; }), type: 'bar', name: 'Real robot', marker: { color: '#f0883e' }, hovertemplate: '%{x}: %{y:.0f}%<extra></extra>' }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Sim-to-Real Gap Across Household Tasks</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Task', gridcolor: '#e5e5e5' }, yaxis: { title: 'Success rate (%)', range: [0, 100], gridcolor: '#e5e5e5' },
            barmode: 'group', height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture18Charts = function () {
        drawBlueprint('plotly-cs231n-18-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture18Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-18-blueprint')) {
        window.renderCs231nLecture18Charts();
    }
})();
