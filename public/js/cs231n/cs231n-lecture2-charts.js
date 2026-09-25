/* ==========================================================================
   CS231N Lecture 2 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-2.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* L1 vs L2 distance contours on a 2D feature grid; rotation slider. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        var N = 60, xs = [], ys = [];
        for (var i = -N; i <= N; i++) { var v = i / N * 3; xs.push(v); ys.push(v); }
        function distGrid(theta, metric) {
            var z = [], ct = Math.cos(theta), st = Math.sin(theta);
            for (var a = 0; a < xs.length; a++) {
                var row = [];
                for (var b = 0; b < xs.length; b++) {
                    var rx = xs[b] * ct - ys[a] * st, ry = xs[b] * st + ys[a] * ct;
                    row.push(metric === 'L1' ? Math.abs(rx) + Math.abs(ry) : Math.sqrt(rx * rx + ry * ry));
                }
                z.push(row);
            }
            return z;
        }
        var data = [
            { type: 'contour', x: xs, y: ys, z: distGrid(0, 'L2'), colorscale: 'Blues', line: { width: 1 }, hovertemplate: 'x %{x:.1f}, y %{y:.1f}<br>d %{z:.2f}<extra></extra>', name: 'L2' },
            { type: 'contour', x: xs, y: ys, z: distGrid(0, 'L1'), colorscale: 'Reds', showscale: false, line: { width: 1 }, visible: false, name: 'L1' }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Distance Metric Geometry: L1 vs L2 Under Rotation</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Feature x1', gridcolor: '#e5e5e5' }, yaxis: { title: 'Feature x2', scaleanchor: 'x', gridcolor: '#e5e5e5' },
            height: 520, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
        var angles = [0, 15, 30, 45, 60, 75, 90];
        var steps = angles.map(function (deg) {
            var th = deg * Math.PI / 180;
            return { label: deg + '\u00B0', method: 'restyle', args: [{ z: [distGrid(th, 'L2'), distGrid(th, 'L1')] }, [0, 1]] };
        });
        layout.sliders = [{ active: 0, x: 0.15, len: 0.7, steps: steps, currentvalue: { prefix: 'Rotation: ' } }];
        layout.updatemenus = [{ type: 'buttons', direction: 'right', x: 0.75, y: 1.12, buttons: [
            { label: 'L2', method: 'restyle', args: [{ visible: [true, false] }] },
            { label: 'L1', method: 'restyle', args: [{ visible: [false, true] }] }
        ] }];
        Plotly.react(el, data, layout, { displayModeBar: false });

        /* Backup listener: re-apply the slider step's restyle args in case the
           slider update method does not fire in some environments. */
        el.on('plotly_sliderupdate', function (ev) {
            var step = el.layout.sliders[0].steps[ev.slider.active];
            if (step && step.args) Plotly.restyle(el, step.args[0], step.args[1]);
        });
    }
    window.renderCs231nLecture2Charts = function () {
        drawBlueprint('plotly-cs231n-2-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture2Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-2-blueprint')) {
        window.renderCs231nLecture2Charts();
    }
})();
