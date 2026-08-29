/* ==========================================================================
   Lecture 2 interactive Plotly charts.
   Loaded by courses/cs229/lecture-2.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-2.html).
   ========================================================================== */

(function () {
    'use strict';

    /* Shared PRNG so the SGD noise is stable across reloads. */
    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
            var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    /* ======================================================================
       Chart: BGD vs SGD optimization trajectories on an elliptical bowl
       J(θ) = 0.5 * (θ1² / a² + θ2² / b²)  — anisotropic curvature so the
       contours are ellipses and the gradient does not point at the minimum.
       ====================================================================== */
    function drawTrajectories(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* 1. Cost surface: elliptical bowl. Wide in θ1, steep in θ2. */
        var a = 3.0, b = 1.0;   // curvature: shallow in θ1, steep in θ2
        function J(t1, t2) { return 0.5 * ((t1 * t1) / (a * a) + (t2 * t2) / (b * b)); }
        function grad(t1, t2) { return [t1 / (a * a), t2 / (b * b)]; }

        /* 2. Contour grid. */
        var xs = [], ys = [], Z = [];
        for (var yv = -4; yv <= 4.0001; yv += 8 / 90) ys.push(Math.round(yv * 1000) / 1000);
        for (var xv = -4; xv <= 4.0001; xv += 8 / 90) xs.push(Math.round(xv * 1000) / 1000);
        ys.forEach(function (yy) {
            Z.push(xs.map(function (xx) { return J(xx, yy); }));
        });

        /* 3. Batch Gradient Descent: full-batch gradient, smooth direct path.
              Start chosen off-axis so the smooth glide along the shallow axis
              is visible (classic GD zig-zag on ill-conditioned bowls). */
        var start = [0.6, 3.4];
        var lr = 0.85;                      // stable: lr < 2 / λmax = 2 / (1/b²) = 2
        var bgd = [start.slice()];
        var p = start.slice();
        for (var s = 0; s < 18; s++) {
            var g = grad(p[0], p[1]);
            p = [p[0] - lr * g[0], p[1] - lr * g[1]];
            bgd.push(p.slice());
        }

        /* 4. Stochastic Gradient Descent: per-sample gradients = true gradient
              + zero-mean noise (variance proportional to curvature spread).
              Same learning rate scale, many more steps, noisy "drunken walk". */
        var rand = mulberry32(42);
        var sgd = [start.slice()];
        var q = start.slice();
        var sgdLr = 0.5;
        for (var t = 0; t < 150; t++) {
            var g2 = grad(q[0], q[1]);
            /* Per-sample gradient noise: larger when far from optimum. */
            var noiseScale = 0.9 * Math.sqrt(J(q[0], q[1]) + 0.08);
            q = [
                q[0] - sgdLr * (g2[0] + noiseScale * (rand() - 0.5)),
                q[1] - sgdLr * (g2[1] + noiseScale * (rand() - 0.5))
            ];
            sgd.push(q.slice());
        }

        /* 5. Arrow annotations along each path. */
        function pathArrows(path, color, every) {
            var ann = [];
            for (var i = 0; i < path.length - 1; i += every) {
                ann.push({
                    ax: path[i][0], ay: path[i][1],
                    x: path[i + 1][0], y: path[i + 1][1],
                    xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                    showarrow: true, arrowhead: 2, arrowsize: 1,
                    arrowwidth: 1.4, arrowcolor: color, opacity: 0.8
                });
            }
            return ann;
        }

        var data = [{
            type: 'contour',
            x: xs, y: ys, z: Z,
            colorscale: 'Viridis', reversescale: true,
            opacity: 0.75,
            line: { smoothing: 0.85 },
            contours: { coloring: 'lines' },
            colorbar: { title: { text: 'J(θ)', side: 'right' }, thickness: 15 },
            hoverinfo: 'skip',
            showlegend: false
        }, {
            x: [0], y: [0], mode: 'markers',
            marker: { color: 'red', size: 11, symbol: 'x', line: { width: 1 } },
            name: 'Optimum θ* = (0, 0)'
        }, {
            x: bgd.map(function (pt) { return pt[0]; }),
            y: bgd.map(function (pt) { return pt[1]; }),
            mode: 'lines+markers', type: 'scatter',
            marker: { symbol: 'circle', size: 5, color: '#1f77b4' },
            line: { color: '#1f77b4', width: 2.5 },
            name: 'Batch GD (smooth, direct)',
            hovertemplate: 'BGD step %{i}<br>θ₁: %{x:.2f}<br>θ₂: %{y:.2f}<extra></extra>'
        }, {
            x: sgd.map(function (pt) { return pt[0]; }),
            y: sgd.map(function (pt) { return pt[1]; }),
            mode: 'lines+markers', type: 'scatter',
            marker: { symbol: 'circle', size: 3.5, color: '#d16a1a' },
            line: { color: '#d16a1a', width: 1.4 },
            name: 'SGD (noisy "drunken walk")',
            hovertemplate: 'SGD step %{i}<br>θ₁: %{x:.2f}<br>θ₂: %{y:.2f}<extra></extra>'
        }, {
            x: [start[0]], y: [start[1]], mode: 'markers',
            marker: { color: 'purple', size: 10, symbol: 'square' },
            name: 'Start θ⁰ = (0.6, 3.4)'
        }];

        Plotly.newPlot(el, data, {
            title: { text: '<b>Batch GD vs. Stochastic GD Trajectories</b><br><sup>Elliptical cost contours: BGD glides to θ*, SGD bounces around it</sup>', x: 0.5 },
            xaxis: { title: 'Parameter θ₁', range: [-4, 4], gridcolor: 'lightgray' },
            yaxis: { title: 'Parameter θ₂', range: [-4, 4], gridcolor: 'lightgray', scaleanchor: 'x', scaleratio: 1 },
            width: 780,
            height: 640,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'lightgray', borderwidth: 1 },
            plot_bgcolor: 'white',
            annotations: pathArrows(bgd, '#1f77b4', 3).concat(pathArrows(sgd, '#d16a1a', 12)),
            margin: { t: 80 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture2Charts = function () {
        drawTrajectories('plotly-bgd-vs-sgd');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture2Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-bgd-vs-sgd')) {
        window.renderLecture2Charts();
    }
})();
