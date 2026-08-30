/* ==========================================================================
   Lecture 3 interactive Plotly charts.
   Loaded by courses/cs229/lecture-3.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-3.html).
   ========================================================================== */

(function () {
    'use strict';

    /* Shared PRNG so the synthetic data matches the Python (seed 42) output. */
    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
            var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    function sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    /* ======================================================================
       Chart 1: Logistic Regression Decision Boundary
       ====================================================================== */
    function drawLogisticBoundary(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* 1. Synthetic classification data (mirrors numpy seed 42 layout). */
        var rand = mulberry32(42);
        var n = 50;
        var theta = [-0.5, 1.2, 1.0]; // [intercept, theta1, theta2]
        var X = [], y = [];
        for (var i = 0; i < n; i++) {
            var x1 = -3 + rand() * 6;
            var x2 = -3 + rand() * 6;
            var p = sigmoid(theta[0] + theta[1] * x1 + theta[2] * x2);
            X.push([x1, x2]);
            y.push(p >= 0.5 ? 1 : 0);
        }

        /* 2. Probability grid for the background contour. */
        var grid = [], probs = [];
        var N = 60;
        for (var gy = -3.5; gy <= 3.5; gy += 7 / N) {
            var row = [];
            for (var gx = -3.5; gx <= 3.5; gx += 7 / N) {
                row.push(sigmoid(theta[0] + theta[1] * gx + theta[2] * gy));
            }
            probs.push(row);
            grid.push(Math.round(gy * 100) / 100);
        }
        var gridX = grid.map(function (v) { return v; });

        /* 3. Decision boundary line: theta^T x = 0. */
        var lineX = [-3.5, 3.5];
        var lineY = lineX.map(function (x1) {
            return -(theta[0] + theta[1] * x1) / theta[2];
        });

        /* 4. Gradient arrows (orthogonal to boundary, direction of ascent). */
        var arrowLen = 0.55;
        var arrowTraces = [];
        var centers = [-2.4, -1.2, 0, 1.2, 2.4];
        centers.forEach(function (cx) {
            centers.forEach(function (cy) {
                arrowTraces.push({
                    x: [cx, cx + theta[1] * arrowLen],
                    y: [cy, cy + theta[2] * arrowLen],
                    mode: 'lines',
                    line: { color: 'rgba(40,40,40,0.55)', width: 1.5 },
                    hoverinfo: 'skip',
                    showlegend: false
                });
                /* Arrowhead */
                var hx = cx + theta[1] * arrowLen;
                var hy = cy + theta[2] * arrowLen;
                arrowTraces.push({
                    x: [hx, hx - 0.12 * theta[1] + 0.08 * theta[2]],
                    y: [hy, hy - 0.12 * theta[2] - 0.08 * theta[1]],
                    mode: 'lines',
                    line: { color: 'rgba(40,40,40,0.55)', width: 1.5 },
                    hoverinfo: 'skip',
                    showlegend: false
                });
                arrowTraces.push({
                    x: [hx, hx - 0.12 * theta[1] - 0.08 * theta[2]],
                    y: [hy, hy - 0.12 * theta[2] + 0.08 * theta[1]],
                    mode: 'lines',
                    line: { color: 'rgba(40,40,40,0.55)', width: 1.5 },
                    hoverinfo: 'skip',
                    showlegend: false
                });
            });
        });

        /* 5. Class-separated points. */
        var pos = X.filter(function (_, i) { return y[i] === 1; });
        var neg = X.filter(function (_, i) { return y[i] === 0; });

        var data = [{
            type: 'contour',
            x: gridX,
            y: grid,
            z: probs,
            colorscale: 'RdBu',
            reversescale: true,
            opacity: 0.45,
            line: { width: 0 },
            contours: { start: 0, end: 1, size: 0.05, showlines: false },
            colorbar: { title: { text: 'h(x) probability', side: 'top' }, thickness: 15 },
            hoverinfo: 'skip',
            showlegend: false
        }, {
            x: lineX, y: lineY, mode: 'lines',
            line: { color: 'black', width: 3, dash: 'dash' },
            name: 'Decision boundary (θᵀx = 0)'
        }].concat(arrowTraces, [{
            x: pos.map(function (p) { return p[0]; }),
            y: pos.map(function (p) { return p[1]; }),
            mode: 'markers',
            marker: { color: 'crimson', size: 10, symbol: 'circle', line: { color: 'black', width: 1 } },
            name: 'Positive class (y = 1)'
        }, {
            x: neg.map(function (p) { return p[0]; }),
            y: neg.map(function (p) { return p[1]; }),
            mode: 'markers',
            marker: { color: 'dodgerblue', size: 11, symbol: 'triangle-up', line: { color: 'black', width: 1 } },
            name: 'Negative class (y = 0)'
        }]);

        Plotly.newPlot(el, data, {
            title: { text: '<b>Logistic Regression Decision Boundary</b><br><sup>Sigmoid probability mapping space</sup>', x: 0.5 },
            xaxis: { title: 'Feature x₁', range: [-3.5, 3.5], gridcolor: 'lightgray' },
            yaxis: { title: 'Feature x₂', range: [-3.5, 3.5], gridcolor: 'lightgray', scaleanchor: 'x', scaleratio: 1 },
            width: 750,
            height: 700,
            plot_bgcolor: 'white',
            responsive: true,
            legend: { y: 0.99, x: 0.01, yanchor: 'top', xanchor: 'left', bgcolor: 'rgba(255,255,255,0.8)' },
            margin: { t: 80 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: Gradient Descent vs Newton's Method
       ====================================================================== */
    function drawOptimizationCompare(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* Objective: f(x) = 0.5*(x1^2 + 5*x2^2), a quadratic bowl. */
        function f(x, y) { return 0.5 * (x * x + 5 * y * y); }
        function grad(x, y) { return [x, 5 * y]; }
        var Hinv = [[1, 0], [0, 0.2]]; // inverse Hessian (constant for this f)

        /* Contour grid. */
        var xs = [], ys = [], Z = [];
        for (var yv = -2; yv <= 2.0001; yv += 4 / 80) {
            ys.push(Math.round(yv * 1000) / 1000);
        }
        for (var xv = -3; xv <= 3.0001; xv += 6 / 100) {
            xs.push(Math.round(xv * 1000) / 1000);
        }
        ys.forEach(function (yy) {
            var row = xs.map(function (xx) { return f(xx, yy); });
            Z.push(row);
        });

        /* Gradient Descent path. */
        var gd = [[2.5, 1.5]];
        var lr = 0.15;
        for (var s = 0; s < 10; s++) {
            var g = grad(gd[gd.length - 1][0], gd[gd.length - 1][1]);
            gd.push([gd[gd.length - 1][0] - lr * g[0], gd[gd.length - 1][1] - lr * g[1]]);
        }

        /* Newton's Method path (converges in ~1 step for a quadratic). */
        var nm = [[2.5, 1.5]];
        for (var t = 0; t < 2; t++) {
            var gn = grad(nm[nm.length - 1][0], nm[nm.length - 1][1]);
            var cur = nm[nm.length - 1];
            var next = [
                cur[0] - (Hinv[0][0] * gn[0] + Hinv[0][1] * gn[1]),
                cur[1] - (Hinv[1][0] * gn[0] + Hinv[1][1] * gn[1])
            ];
            nm.push(next);
            if (Math.abs(next[0]) < 1e-6 && Math.abs(next[1]) < 1e-6) break;
        }

        /* Arrow annotations along each path. */
        function pathArrows(path, color) {
            var ann = [];
            for (var i = 0; i < path.length - 1; i++) {
                ann.push({
                    ax: path[i][0], ay: path[i][1],
                    x: path[i + 1][0], y: path[i + 1][1],
                    xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                    showarrow: true, arrowhead: 2, arrowsize: 1,
                    arrowwidth: 1.5, arrowcolor: color, opacity: 0.85
                });
            }
            return ann;
        }

        var data = [{
            type: 'contour',
            x: xs, y: ys, z: Z,
            colorscale: 'Viridis',
            opacity: 0.7,
            line: { smoothing: 0.85 },
            contours: { coloring: 'lines' },
            colorbar: { title: { text: 'f(x)', side: 'right' } },
            name: 'Objective contours'
        }, {
            x: [0], y: [0], mode: 'markers',
            marker: { color: 'red', size: 10, symbol: 'circle' },
            name: 'Global minimum'
        }, {
            x: gd.map(function (p) { return p[0]; }),
            y: gd.map(function (p) { return p[1]; }),
            mode: 'lines+markers',
            marker: { symbol: 'circle', size: 5, color: 'blue' },
            line: { color: 'blue', width: 2 },
            name: 'Gradient Descent'
        }, {
            x: nm.map(function (p) { return p[0]; }),
            y: nm.map(function (p) { return p[1]; }),
            mode: 'lines+markers',
            marker: { symbol: 'circle', size: 5, color: 'green' },
            line: { color: 'green', width: 3 },
            name: "Newton's Method"
        }, {
            x: [2.5], y: [1.5], mode: 'markers',
            marker: { color: 'purple', size: 10, symbol: 'square' },
            name: 'Starting point'
        }];

        var layout = {
            title: { text: "<b>Gradient Descent vs. Newton's Method</b>", x: 0.5, font: { size: 18 } },
            xaxis: { title: 'Parameter x₁', range: [-3, 3], gridcolor: 'lightgray' },
            yaxis: { title: 'Parameter x₂', range: [-2, 2], gridcolor: 'lightgray', scaleanchor: 'x', scaleratio: 1 },
            height: 700,
            width: 800,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.7)', bordercolor: 'lightgray', borderwidth: 1 },
            plot_bgcolor: 'white',
            annotations: pathArrows(gd, 'blue').concat(pathArrows(nm, 'green')),
            margin: { t: 60 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture3Charts = function () {
        drawLogisticBoundary('plotly-logistic-boundary');
        drawOptimizationCompare('plotly-optimization-compare');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture3Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-logistic-boundary')) {
        window.renderLecture3Charts();
    }
})();
