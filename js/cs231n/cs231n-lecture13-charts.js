/* ==========================================================================
   CS231N Lecture 13 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-13.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* VAE latent slice: cluster separation vs reconstruction under beta. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        function prng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
        var rnd = prng(55);
        var betas = [0, 0.5, 1, 2, 4, 8];
        var classes = 5, per = 20;
        var centers = [[-4, -3], [4, -3], [0, 2], [-5, 4], [5, 4]];
        var data = [];
        for (var c = 0; c < classes; c++) {
            var xs = [], ys = [];
            for (var k = 0; k < per; k++) {
                xs.push(centers[c][0] + (rnd() - 0.5) * 1.6);
                ys.push(centers[c][1] + (rnd() - 0.5) * 1.6);
            }
            data.push({ x: xs, y: ys, mode: 'markers', name: 'class ' + c, marker: { size: 7, opacity: 0.8 } });
        }
        function layoutFor(beta) {
            var pull = Math.max(0.25, 1 - beta / 10);
            return {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                title: { text: '<b>VAE Latent Slice at \u03B2 = ' + beta + '</b>', x: 0.5, y: 0.98, font: { size: 15 } },
                xaxis: { title: 'Z1', range: [-6 * pull - 1.5, 6 * pull + 1.5], zeroline: true, gridcolor: '#e5e5e5' },
                yaxis: { title: 'Z2', range: [-6 * pull - 1.5, 6 * pull + 1.5], zeroline: true, scaleanchor: 'x', gridcolor: '#e5e5e5' },
                height: 520, responsive: true, showlegend: false,
                margin: { t: 90, b: 55, l: 70, r: 50 }
            };
        }
        Plotly.newPlot(el, data, layoutFor(0), { displayModeBar: false });
        var steps = betas.map(function (b) {
            return { label: String(b), method: 'relayout', args: [layoutFor(b)] };
        });
        var layout = layoutFor(0);
        layout.sliders = [{ active: 0, x: 0.15, len: 0.7, steps: steps, currentvalue: { prefix: '\u03B2 = ' } }];
        Plotly.react(el, data, layout, { displayModeBar: false });

        /* Backup listener: re-apply the slider step's relayout args in case the
           slider update method does not fire in some environments. */
        el.on('plotly_sliderupdate', function (ev) {
            var step = el.layout.sliders[0].steps[ev.slider.active];
            if (step && step.args) Plotly.relayout(el, step.args[0]);
        });
    }
    window.renderCs231nLecture13Charts = function () {
        drawBlueprint('plotly-cs231n-13-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture13Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-13-blueprint')) {
        window.renderCs231nLecture13Charts();
    }
})();
