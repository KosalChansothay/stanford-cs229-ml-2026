/* ==========================================================================
   Lecture 5 interactive Plotly charts.
   Loaded by courses/cs229/lecture-5.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-5.html).
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

    /* Gaussian log-density under mean mu and 2x2 covariance Sigma. */
    function gaussLog(x, y, mu, Sigma) {
        var dx = x - mu[0], dy = y - mu[1];
        var det = Sigma[0][0] * Sigma[1][1] - Sigma[0][1] * Sigma[1][0];
        var i00 = Sigma[1][1] / det, i01 = -Sigma[0][1] / det;
        var i10 = -Sigma[1][0] / det, i11 = Sigma[0][0] / det;
        var quad = dx * (i00 * dx + i01 * dy) + dy * (i10 * dx + i11 * dy);
        return -0.5 * quad - 0.5 * Math.log(det);
    }

    /* ======================================================================
       Chart: GDA vs QDA decision boundaries (side-by-side subplots)
       Left: shared Sigma -> linear boundary. Right: per-class Sigma -> quadratic.
       ====================================================================== */
    function drawGdaQda(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = mulberry32(5);

        /* Class parameters. mu0/mu1 shared by both models. */
        var mu0 = [-1.8, -1.0], mu1 = [1.8, 1.0];
        var phi = 0.5; // class prior

        /* GDA: shared covariance (same for both classes). */
        var SigmaShared = [[2.2, 0.9], [0.9, 1.0]];
        /* QDA: different covariances per class (rotated differently). */
        var Sigma0 = [[2.4, 1.3], [1.3, 0.8]];
        var Sigma1 = [[1.0, -0.9], [-0.9, 2.2]];

        /* 1. Synthetic samples from each class (anisotropic Gaussians). */
        function sample(mu, Sigma, n) {
            /* Cholesky decomposition of 2x2 SPD matrix. */
            var L00 = Math.sqrt(Sigma[0][0]);
            var L10 = Sigma[1][0] / L00;
            var L11 = Math.sqrt(Sigma[1][1] - L10 * L10);
            var pts = [];
            for (var i = 0; i < n; i++) {
                var z1 = rand() * 2 - 1 + (rand() - 0.5) * 1.2;
                var z2 = rand() * 2 - 1 + (rand() - 0.5) * 1.2;
                pts.push([
                    mu[0] + L00 * z1 + 0 * z2,
                    mu[1] + L10 * z1 + L11 * z2
                ]);
            }
            return pts;
        }
        var pts0 = sample(mu0, Sigma0, 60);
        var pts1 = sample(mu1, Sigma1, 60);

        /* 2. Boundary grids: posterior log-odds = 0. */
        var N = 90, xs = [], ys = [];
        for (var yv = -5; yv <= 5.0001; yv += 10 / N) ys.push(Math.round(yv * 1000) / 1000);
        for (var xv = -5; xv <= 5.0001; xv += 10 / N) xs.push(Math.round(xv * 1000) / 1000);
        var prior = Math.log(phi / (1 - phi));

        function logOdds(x, y, S0, S1) {
            return gaussLog(x, y, mu1, S1) - gaussLog(x, y, mu0, S0) + prior;
        }

        var Zgda = [], Zqda = [];
        ys.forEach(function (yy) {
            var r1 = [], r2 = [];
            xs.forEach(function (xx) {
                r1.push(logOdds(xx, yy, SigmaShared, SigmaShared));
                r2.push(logOdds(xx, yy, Sigma0, Sigma1));
            });
            Zgda.push(r1);
            Zqda.push(r2);
        });

        /* 3. Zero-contour extraction (marching-squares-lite): collect segments
              where the sign changes, draw as scattered line segments. */
        function zeroSegments(Z) {
            var segs = { x: [], y: [] };
            for (var j = 0; j < N; j++) {
                for (var i = 0; i < N; i++) {
                    var a = Z[j][i], b = Z[j][i + 1], c = Z[j + 1][i], d = Z[j + 1][i + 1];
                    var x0 = xs[i], x1 = xs[i + 1], y0 = ys[j], y1 = ys[j + 1];
                    var edges = [];
                    if ((a > 0) !== (b > 0)) edges.push([x0 + (0 - a) / (b - a) * (x1 - x0), y0]);
                    if ((a > 0) !== (c > 0)) edges.push([x0, y0 + (0 - a) / (c - a) * (y1 - y0)]);
                    if ((b > 0) !== (d > 0)) edges.push([x1, y0 + (0 - b) / (d - b) * (y1 - y0)]);
                    if ((c > 0) !== (d > 0)) edges.push([x0 + (0 - c) / (d - c) * (x1 - x0), y1]);
                    if (edges.length >= 2) {
                        segs.x.push(edges[0][0], edges[1][0], null);
                        segs.y.push(edges[0][1], edges[1][1], null);
                    }
                }
            }
            return segs;
        }
        var segGda = zeroSegments(Zgda);
        var segQda = zeroSegments(Zqda);

        var data = [
            /* Left subplot: GDA */
            {
                type: 'contour', xaxis: 'x', yaxis: 'y',
                x: xs, y: ys, z: Zgda,
                colorscale: 'RdBu', reversescale: true, opacity: 0.4,
                line: { width: 0 }, contours: { start: -4, end: 4, size: 0.8, showlines: false },
                hovertemplate: 'GDA log-odds: %{z:.2f}<extra></extra>',
                colorbar: { x: 0.46, thickness: 12, title: { text: 'log-odds', side: 'top' }, len: 0.9 }
            },
            {
                x: segGda.x, y: segGda.y, mode: 'lines', type: 'scatter',
                xaxis: 'x', yaxis: 'y',
                line: { color: 'black', width: 3.5 },
                name: 'GDA boundary (linear)',
                hoverinfo: 'skip'
            },
            /* Right subplot: QDA */
            {
                type: 'contour', xaxis: 'x2', yaxis: 'y2',
                x: xs, y: ys, z: Zqda,
                colorscale: 'RdBu', reversescale: true, opacity: 0.4,
                line: { width: 0 }, contours: { start: -4, end: 4, size: 0.8, showlines: false },
                hovertemplate: 'QDA log-odds: %{z:.2f}<extra></extra>',
                showscale: false
            },
            {
                x: segQda.x, y: segQda.y, mode: 'lines', type: 'scatter',
                xaxis: 'x2', yaxis: 'y2',
                line: { color: 'black', width: 3.5, dash: 'dot' },
                name: 'QDA boundary (quadratic)',
                hoverinfo: 'skip'
            },
            /* Class points on both subplots. */
            {
                x: pts0.map(function (p) { return p[0]; }),
                y: pts0.map(function (p) { return p[1]; }),
                mode: 'markers', type: 'scatter', xaxis: 'x', yaxis: 'y',
                marker: { color: 'dodgerblue', size: 7, symbol: 'circle', line: { color: 'black', width: 0.8 } },
                name: 'Class 0 (y = 0)'
            },
            {
                x: pts1.map(function (p) { return p[0]; }),
                y: pts1.map(function (p) { return p[1]; }),
                mode: 'markers', type: 'scatter', xaxis: 'x', yaxis: 'y',
                marker: { color: 'crimson', size: 7, symbol: 'triangle-up', line: { color: 'black', width: 0.8 } },
                name: 'Class 1 (y = 1)'
            },
            {
                x: pts0.map(function (p) { return p[0]; }),
                y: pts0.map(function (p) { return p[1]; }),
                mode: 'markers', type: 'scatter', xaxis: 'x2', yaxis: 'y2',
                marker: { color: 'dodgerblue', size: 7, symbol: 'circle', line: { color: 'black', width: 0.8 } },
                name: 'Class 0 (y = 0)', showlegend: false
            },
            {
                x: pts1.map(function (p) { return p[0]; }),
                y: pts1.map(function (p) { return p[1]; }),
                mode: 'markers', type: 'scatter', xaxis: 'x2', yaxis: 'y2',
                marker: { color: 'crimson', size: 7, symbol: 'triangle-up', line: { color: 'black', width: 0.8 } },
                name: 'Class 1 (y = 1)', showlegend: false
            }
        ];

        var axisCommon = {
            range: [-5, 5], gridcolor: 'lightgray', zeroline: true, zerolinecolor: '#999',
            zerolinewidth: 1
        };

        Plotly.newPlot(el, data, {
            title: { text: '<b>GDA vs. QDA: Shared vs. Per-Class Covariance</b><br><sup>Shared Σ ⇒ straight boundary (quadratic terms cancel) · Per-class Σ ⇒ curved quadratic boundary</sup>', x: 0.5 },
            xaxis: Object.assign({ title: 'x₁ (GDA, shared Σ)', domain: [0, 0.44] }, axisCommon),
            yaxis: Object.assign({ title: 'x₂', scaleanchor: 'x', scaleratio: 1 }, axisCommon),
            xaxis2: Object.assign({ title: 'x₁ (QDA, per-class Σ)', domain: [0.56, 1] }, axisCommon),
            yaxis2: Object.assign({ scaleanchor: 'x', scaleratio: 1 }, axisCommon),
            width: 900,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.5, y: -0.14, xanchor: 'center', orientation: 'h', bgcolor: 'rgba(255,255,255,0.8)' },
            plot_bgcolor: 'white',
            margin: { t: 90, b: 90 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture5Charts = function () {
        drawGdaQda('plotly-gda-vs-qda');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture5Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-gda-vs-qda')) {
        window.renderLecture5Charts();
    }
})();
