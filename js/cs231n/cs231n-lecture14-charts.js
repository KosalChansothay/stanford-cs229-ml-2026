/* ==========================================================================
   CS231N Lecture 14 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-14.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Diffusion particle flow noise->data under different guidance scales. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(77);
        var modes = [[-3, 0], [3, 0], [0, 3], [0, -3]];
        var np = 60, ts = [0, 0.25, 0.5, 0.75, 1];
        /* Same particle seeds per mode for every w: guidance only moves the
           fixed particles, so slider steps stay deterministic and comparable. */
        var seeds = [];
        for (var m = 0; m < modes.length; m++) {
            var sx = [], sy = [];
            for (var p = 0; p < np / modes.length; p++) {
                sx.push((rnd() - 0.5) * 14);
                sy.push((rnd() - 0.5) * 14);
            }
            seeds.push({ sx: sx, sy: sy });
        }
        function snap(w) {
            var traces = [];
            for (var m = 0; m < modes.length; m++) {
                var xs = [], ys = [];
                var focus = 0.35 + 0.12 * w;
                for (var p = 0; p < np / modes.length; p++) {
                    var x0 = seeds[m].sx[p], y0 = seeds[m].sy[p];
                    var xT = modes[m][0] * focus + x0 * (1 - focus);
                    var yT = modes[m][1] * focus + y0 * (1 - focus);
                    xs.push(xT); ys.push(yT);
                }
                traces.push({ x: xs, y: ys, mode: 'markers', name: 'mode ' + m, marker: { size: 5, opacity: 0.7 } });
            }
            return traces;
        }
        function layoutFor(w) {
            return {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                title: { text: '<b>Particle Flow at Guidance Scale w = ' + w + '</b>', x: 0.5, y: 0.98, font: { size: 15 } },
                xaxis: { title: 'Z1', range: [-8, 8], zeroline: true, gridcolor: '#e5e5e5' }, yaxis: { title: 'Z2', range: [-8, 8], zeroline: true, scaleanchor: 'x', gridcolor: '#e5e5e5' },
                height: 520, responsive: true, showlegend: false,
                margin: { t: 90, b: 55, l: 70, r: 50 }
            };
        }
        var ws = [0, 1, 2, 4, 8];
        var data = snap(0), layout = layoutFor(0);
        var steps = ws.map(function (w) {
            var s = snap(w);
            return { label: String(w), method: 'update', args: [{ x: s.map(function (t) { return t.x; }), y: s.map(function (t) { return t.y; }) }, layoutFor(w)] };
        });
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
        layout.sliders = [{ active: 0, x: 0.15, len: 0.7, steps: steps, currentvalue: { prefix: 'w = ' } }];
        Plotly.react(el, data, layout, { displayModeBar: false });

        /* Backup listener: re-apply the slider step's update args in case the
           slider update method does not fire in some environments. */
        el.on('plotly_sliderupdate', function (ev) {
            var step = el.layout.sliders[0].steps[ev.slider.active];
            if (step && step.args) Plotly.update(el, step.args[0], step.args[1] || {}, step.args[2] || []);
        });
    }
    window.renderCs231nLecture14Charts = function () {
        drawBlueprint('plotly-cs231n-14-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture14Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-14-blueprint')) {
        window.renderCs231nLecture14Charts();
    }
})();
