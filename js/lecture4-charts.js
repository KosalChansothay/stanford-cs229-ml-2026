/* ==========================================================================
   Lecture 4 interactive Plotly charts.
   Loaded by courses/cs229/lecture-4.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-4.html).
   ========================================================================== */

(function () {
    'use strict';

    /* Shared PRNG for stable synthetic data. */
    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
            var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    /* ======================================================================
       Chart 1: The GLM Parameter Crank
       A pipeline diagram: x -> (dot theta) -> eta -> g(eta) -> h(x).
       A draggable slider (Plotly slider) moves the input x and every stage
       updates, showing how the "crank" turns.
       ====================================================================== */
    function drawGlmCrank(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* Bernoulli GLM: eta = theta0 + theta1 * x, h = sigmoid(eta). */
        var theta0 = -3, theta1 = 1.2;

        function stages(x) {
            var eta = theta0 + theta1 * x;
            var h = sigmoid(eta);
            return { eta: eta, h: h };
        }

        function frame(x) {
            var s = stages(x);
            return {
                annotations: [
                    { text: '<b>Features</b><br>x = ' + x.toFixed(2), x: 0.5, y: 0.5, xref: 'x', yref: 'y', showarrow: false, font: { size: 14, color: '#173f5f' }, align: 'center' },
                    { text: '<b>Weights</b><br>θᵀx = ' + s.eta.toFixed(2), x: 1.5, y: 0.5, xref: 'x', yref: 'y', showarrow: false, font: { size: 14, color: '#173f5f' }, align: 'center' },
                    { text: '<b>Natural param</b><br>η = ' + s.eta.toFixed(2), x: 2.5, y: 0.5, xref: 'x', yref: 'y', showarrow: false, font: { size: 14, color: '#173f5f' }, align: 'center' },
                    { text: '<b>Response fn</b><br>g(η) = sigmoid', x: 3.5, y: 0.5, xref: 'x', yref: 'y', showarrow: false, font: { size: 14, color: '#173f5f' }, align: 'center' },
                    { text: '<b>Prediction</b><br>h(x) = ' + s.h.toFixed(3), x: 4.5, y: 0.5, xref: 'x', yref: 'y', showarrow: false, font: { size: 14, color: '#7a1f5c' }, align: 'center' }
                ],
                shapes: [
                    /* Connecting arrows between stages. */
                    { type: 'line', x0: 0.85, x1: 1.15, y0: 0.5, y1: 0.5, xref: 'x', yref: 'y', line: { color: '#8a9798', width: 2 } },
                    { type: 'line', x0: 1.85, x1: 2.15, y0: 0.5, y1: 0.5, xref: 'x', yref: 'y', line: { color: '#8a9798', width: 2 } },
                    { type: 'line', x0: 2.85, x1: 3.15, y0: 0.5, y1: 0.5, xref: 'x', yref: 'y', line: { color: '#8a9798', width: 2 } },
                    { type: 'line', x0: 3.85, x1: 4.15, y0: 0.5, y1: 0.5, xref: 'x', yref: 'y', line: { color: '#8a9798', width: 2 } },
                    /* Stage boxes. */
                    { type: 'rect', x0: 0, x1: 1, y0: 0.3, y1: 0.7, xref: 'x', yref: 'y', line: { color: '#173f5f', width: 2 }, fillcolor: 'rgba(40,107,130,0.08)' },
                    { type: 'rect', x0: 1, x1: 2, y0: 0.3, y1: 0.7, xref: 'x', yref: 'y', line: { color: '#173f5f', width: 2 }, fillcolor: 'rgba(40,107,130,0.08)' },
                    { type: 'rect', x0: 2, x1: 3, y0: 0.3, y1: 0.7, xref: 'x', yref: 'y', line: { color: '#173f5f', width: 2 }, fillcolor: 'rgba(40,107,130,0.08)' },
                    { type: 'rect', x0: 3, x1: 4, y0: 0.3, y1: 0.7, xref: 'x', yref: 'y', line: { color: '#173f5f', width: 2 }, fillcolor: 'rgba(40,107,130,0.08)' },
                    { type: 'rect', x0: 4, x1: 5, y0: 0.3, y1: 0.7, xref: 'x', yref: 'y', line: { color: '#7a1f5c', width: 2 }, fillcolor: 'rgba(122,31,92,0.10)' }
                ]
            };
        }

        var x0 = 3;
        var s0 = stages(x0);

        var data = [{
            /* Sigmoid curve in the background of the response stage. */
            x: [-6, 6], y: [0, 1], mode: 'lines', type: 'scatter',
            line: { color: 'rgba(0,0,0,0)', width: 0 }, hoverinfo: 'skip', showlegend: false
        }];

        var sliderSteps = [];
        for (var xv = 0; xv <= 6; xv += 0.5) {
            (function (xv) {
                sliderSteps.push({
                    method: 'update',
                    label: xv.toFixed(1),
                    args: [{}, frame(xv)]
                });
            })(xv);
        }

        Plotly.newPlot(el, data, {
            title: { text: '<b>The GLM Parameter Crank</b><br><sup>Drag the slider: watch x flow through the pipeline to the prediction</sup>', x: 0.5 },
            xaxis: { range: [0, 5], visible: false },
            yaxis: { range: [0, 1], visible: false },
            width: 820,
            height: 340,
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: { t: 80, b: 70 },
            sliders: [{
                active: 6,
                currentvalue: { prefix: 'input x = ', font: { size: 13 } },
                pad: { t: 30 },
                steps: sliderSteps
            }]
        }, { displayModeBar: false })
            .then(function (gd) { Plotly.update(gd, {}, frame(x0)); });
    }

    /* ======================================================================
       Chart 2: Softmax Geometry in 2D Feature Space
       Four classes (Cat, Dog, Car, Bus) with linear scores; background
       heatmap of the max-class probability + decision regions + points.
       ====================================================================== */
    function drawSoftmaxGeometry(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = mulberry32(11);

        /* Four class weight vectors (bias folded into 3D x). */
        var classes = [
            { name: 'Cat', color: '#d16a1a', w: [2.0, 1.0, -6] },
            { name: 'Dog', color: '#1f77b4', w: [-1.5, 2.0, -4] },
            { name: 'Car', color: '#2ca02c', w: [1.0, -2.0, -2] },
            { name: 'Bus', color: '#7a1f5c', w: [-2.0, -1.0, 4] }
        ];

        function softmax(x1, x2) {
            var scores = classes.map(function (c) { return c.w[0] * x1 + c.w[1] * x2 + c.w[2]; });
            var m = Math.max.apply(null, scores);
            var exps = scores.map(function (s) { return Math.exp(s - m); });
            var sum = exps.reduce(function (a, b) { return a + b; }, 0);
            return exps.map(function (e) { return e / sum; });
        }

        /* 1. Probability grid: max-class probability (confidence). */
        var N = 70, xs = [], ys = [], Zconf = [], Zcls = [];
        for (var yv = -6; yv <= 6.0001; yv += 12 / N) ys.push(Math.round(yv * 1000) / 1000);
        for (var xv = -6; xv <= 6.0001; xv += 12 / N) xs.push(Math.round(xv * 1000) / 1000);
        ys.forEach(function (yy) {
            var rowC = [], rowK = [];
            xs.forEach(function (xx) {
                var p = softmax(xx, yy);
                var maxP = Math.max.apply(null, p);
                var arg = p.indexOf(maxP);
                rowC.push(maxP);
                rowK.push(arg);
            });
            Zconf.push(rowC);
            Zcls.push(rowK);
        });

        /* 2. Synthetic points sampled around each class's mean direction. */
        var traces = [{
            type: 'contour',
            x: xs, y: ys, z: Zconf,
            colorscale: 'RdYlGn', reversescale: true,
            opacity: 0.55,
            line: { width: 0 },
            contours: { start: 0.25, end: 1, size: 0.05, showlines: false },
            colorbar: { title: { text: 'max p(y=j|x)', side: 'top' }, thickness: 14, tickformat: '.1f' },
            hovertemplate: 'x₁: %{x:.1f}<br>x₂: %{y:.1f}<br>confidence: %{z:.2f}<extra></extra>',
            showlegend: false
        }];

        var symbols = ['circle', 'triangle-up', 'square', 'diamond'];
        classes.forEach(function (c, k) {
            var pts = [];
            var meanDir = [c.w[0], c.w[1]];
            var norm = Math.sqrt(meanDir[0] * meanDir[0] + meanDir[1] * meanDir[1]);
            var cx = meanDir[0] / norm * 3.2, cy = meanDir[1] / norm * 3.2;
            for (var i = 0; i < 22; i++) {
                pts.push([cx + (rand() - 0.5) * 3.4, cy + (rand() - 0.5) * 3.4]);
            }
            traces.push({
                x: pts.map(function (p) { return p[0]; }),
                y: pts.map(function (p) { return p[1]; }),
                mode: 'markers', type: 'scatter', name: c.name,
                marker: { color: c.color, size: 9, symbol: symbols[k], line: { color: 'black', width: 1 } },
                hovertemplate: c.name + '<br>x₁: %{x:.1f}<br>x₂: %{y:.1f}<extra></extra>'
            });
        });

        Plotly.newPlot(el, traces, {
            title: { text: '<b>Softmax Geometry: 4-Class Decision Regions</b><br><sup>Green = confident (deep in a region), red = contentious (near boundaries)</sup>', x: 0.5 },
            xaxis: { title: 'Feature x₁', range: [-6, 6], gridcolor: 'lightgray' },
            yaxis: { title: 'Feature x₂', range: [-6, 6], gridcolor: 'lightgray', scaleanchor: 'x', scaleratio: 1 },
            width: 780,
            height: 640,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            plot_bgcolor: 'white',
            margin: { t: 80 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture4Charts = function () {
        drawGlmCrank('plotly-glm-crank');
        drawSoftmaxGeometry('plotly-softmax-geometry');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture4Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-glm-crank')) {
        window.renderLecture4Charts();
    }
})();
