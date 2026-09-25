/* ==========================================================================
   CS231N Lecture 3 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-3.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* 2D loss landscape + optimizer trajectories (SGD vs Momentum vs Adam). */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function loss(x, y, cond) { return 0.5 * (cond * x * x + y * y) + 0.4 * Math.sin(4 * x) * Math.cos(3 * y); }
        var cond = 4;
        var xs = [], ys = [], z = [];
        for (var i = -30; i <= 30; i++) {
            xs.push(i / 10); var row = [];
            for (var j = -30; j <= 30; j++) { if (i === -30) ys.push(j / 10); row.push(loss(i / 10, j / 10, cond)); }
            z.push(row);
        }
        function runOpt(gx, gy, lr, beta, kind) {
            var px = gx, py = gy, mx = 0, my = 0, vx = 0, vy = 0, path = [[px, py]];
            for (var t = 0; t < 60; t++) {
                var e = 0.05;
                var dx = (loss(px + e, py, cond) - loss(px - e, py, cond)) / (2 * e);
                var dy = (loss(px, py + e, cond) - loss(px, py - e, cond)) / (2 * e);
                mx = (beta || 0) * mx + dx; my = (beta || 0) * my + dy;
                if (kind === 'adam') { vx = 0.9 * vx + 0.1 * dx * dx; vy = 0.9 * vy + 0.1 * dy * dy; }
                var ux = kind === 'momentum' ? -lr * mx : kind === 'adam' ? -lr * mx / (Math.sqrt(vx) + 1e-8) : -lr * dx;
                var uy = kind === 'momentum' ? -lr * my : kind === 'adam' ? -lr * my / (Math.sqrt(vy) + 1e-8) : -lr * dy;
                px += ux; py += uy; path.push([px, py]);
            }
            return path;
        }
        var paths = { sgd: runOpt(-2.5, 2.5, 0.08, 0, 'sgd'), mom: runOpt(-2.5, 2.5, 0.04, 0.9, 'momentum'), adam: runOpt(-2.5, 2.5, 0.25, 0, 'adam') };
        function traceOf(p, name, color) {
            return { x: p.map(function (q) { return q[0]; }), y: p.map(function (q) { return q[1]; }), mode: 'lines+markers', name: name, line: { color: color, width: 2.5 }, marker: { size: 4 }, hovertemplate: name + '<br>(%{x:.2f}, %{y:.2f})<extra></extra>' };
        }
        var data = [
            { type: 'contour', x: xs, y: ys, z: z, colorscale: 'RdBu', line: { width: 0.5 }, hovertemplate: 'L(%{x:.1f}, %{y:.1f}) = %{z:.2f}<extra></extra>', showscale: false },
            traceOf(paths.sgd, 'SGD', '#ff7b72'),
            traceOf(paths.mom, 'Momentum', '#79c0ff'),
            traceOf(paths.adam, 'Adam', '#7ee787')
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Optimizer Trajectories on an Anisotropic Loss Surface</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'w1', gridcolor: '#e5e5e5' }, yaxis: { title: 'w2', gridcolor: '#e5e5e5' }, height: 520, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture3Charts = function () {
        drawBlueprint('plotly-cs231n-3-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture3Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-3-blueprint')) {
        window.renderCs231nLecture3Charts();
    }
})();
