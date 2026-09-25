/* ==========================================================================
   CS231N Lecture 12 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-12.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Contrastive embedding clusters forming over epochs; temperature effect. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var classes = 4, per = 25, epochs = [0, 10, 25, 50, 100];
        var centers = [[0, 0], [6, 0], [3, 5], [9, 5]];
        /* Precompute one deterministic snapshot per epoch so slider steps are
           stable and reproducible across reloads (PRNG state advances once). */
        var snaps = epochs.map(function (e) {
            var rndE = prng(97 + e);
            var spread = Math.max(0.05, 2.2 * Math.exp(-e / 30));
            var traces = [];
            for (var c = 0; c < classes; c++) {
                var xs = [], ys = [];
                for (var k = 0; k < per; k++) {
                    xs.push(centers[c][0] + (rndE() - 0.5) * 2 * spread);
                    ys.push(centers[c][1] + (rndE() - 0.5) * 2 * spread);
                }
                traces.push({ x: xs, y: ys, mode: 'markers', name: 'class ' + c, marker: { size: 6, opacity: 0.75 } });
            }
            return traces;
        });
        function snap(epoch) { return snaps[epochs.indexOf(epoch)]; }
        var data = snap(0);
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>Embedding Space: Clusters Form Under Contrastive Loss</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'PC 1', zeroline: false, gridcolor: '#e5e5e5' }, yaxis: { title: 'PC 2', zeroline: false, scaleanchor: 'x', gridcolor: '#e5e5e5' },
            height: 520, responsive: true, showlegend: false,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
        var steps = epochs.map(function (e) {
            return { label: String(e), method: 'update', args: [{ x: snap(e).map(function (t) { return t.x; }), y: snap(e).map(function (t) { return t.y; }) }] };
        });
        layout.sliders = [{ active: 4, x: 0.15, len: 0.7, steps: steps, currentvalue: { prefix: 'Epoch: ' } }];
        Plotly.react(el, data, layout, { displayModeBar: false });

        /* Backup listener: re-apply the slider step's update args in case the
           slider update method does not fire in some environments. */
        el.on('plotly_sliderupdate', function (ev) {
            var step = el.layout.sliders[0].steps[ev.slider.active];
            if (step && step.args) Plotly.update(el, step.args[0], step.args[1] || {}, step.args[2] || []);
        });
    }
    window.renderCs231nLecture12Charts = function () {
        drawBlueprint('plotly-cs231n-12-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture12Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-12-blueprint')) {
        window.renderCs231nLecture12Charts();
    }
})();
