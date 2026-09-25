/* ==========================================================================
   Lecture 10 interactive Plotly charts.
   Loaded by courses/cs229/lecture-10.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-10.html).
   ========================================================================== */

(function () {
    'use strict';

    /* ======================================================================
       Chart 1: The Geometry of the EM Step
       Same construction as Lecture 9's EM bound (green true log-likelihood,
       orange tangent lower bound) with a slider climbing the surface.
       ====================================================================== */
    function drawEmGeometry(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* True multi-modal log-likelihood and its derivative. */
        function ell(t) {
            return 2.2 * Math.exp(-Math.pow(t - 1.2, 2) / 0.9)
                + 1.6 * Math.exp(-Math.pow(t - 4.2, 2) / 1.8)
                + 0.15 * t - 1.2;
        }
        function dell(t) {
            var e1 = 2.2 * Math.exp(-Math.pow(t - 1.2, 2) / 0.9);
            var e2 = 1.6 * Math.exp(-Math.pow(t - 4.2, 2) / 1.8);
            return e1 * (-2 * (t - 1.2) / 0.9) + e2 * (-2 * (t - 4.2) / 1.8) + 0.15;
        }

        /* EM trajectory: damped ascent (maximizer of the tangent bound). */
        var thetas = [0.2];
        for (var it = 0; it < 5; it++) {
            var u = thetas[thetas.length - 1];
            for (var s = 0; s < 30; s++) {
                u = u + 0.35 * dell(u);
                if (u > 6.5) u = 6.5;
            }
            thetas.push(u);
        }

        var xs = [];
        for (var xv = -0.2; xv <= 6.5001; xv += 6.7 / 300) xs.push(xv);
        var ellY = xs.map(ell);

        function boundTrace(t) {
            var slope = dell(t);
            var y0 = ell(t);
            var bx = [-0.2, 6.5];
            var by = bx.map(function (u) { return y0 + slope * (u - t); });
            return {
                x: bx, y: by, mode: 'lines', type: 'scatter',
                line: { color: '#d62728', width: 2.5 },
                name: 'Lower bound L(Q_t, θ) — tangent at θ = ' + t.toFixed(2),
                hovertemplate: 'bound: %{y:.2f}<extra></extra>'
            };
        }

        /* Always emit 4 traces (stable indices for slider restyle). */
        function frame(step) {
            var t = thetas[step];
            var traces = [
                {
                    x: xs, y: ellY, mode: 'lines', type: 'scatter',
                    line: { color: '#2ca02c', width: 3 },
                    name: 'True log-likelihood l(θ)',
                    hovertemplate: 'θ: %{x:.2f}<br>l(θ): %{y:.2f}<extra></extra>'
                },
                boundTrace(t),
                {
                    x: [t], y: [ell(t)], mode: 'markers', type: 'scatter',
                    marker: { color: '#7a1f5c', size: 12, symbol: 'circle' },
                    name: 'θ⁽ᵗ⁾ = ' + t.toFixed(2),
                    hovertemplate: 'current θ: %{x:.2f}<br>l: %{y:.2f}<extra>tight bound (Jensen equality)</extra>'
                },
                {
                    x: step > 0 ? thetas.slice(0, step + 1) : [],
                    y: step > 0 ? thetas.slice(0, step + 1).map(ell) : [],
                    mode: 'lines+markers', type: 'scatter',
                    line: { color: '#7a1f5c', width: 1.5, dash: 'dot' },
                    marker: { size: 6, symbol: 'circle-open' },
                    name: 'EM trajectory (monotonically increasing)',
                    hoverinfo: 'skip'
                }
            ];
            return traces;
        }

        var steps = thetas.map(function (_, s) {
            var f = frame(s);
            var attr = { x: [], y: [], line: [], marker: [] };
            f.forEach(function (tr) {
                attr.x.push(tr.x);
                attr.y.push(tr.y);
                attr.line.push(tr.line || null);
                attr.marker.push(tr.marker || null);
            });
            return { method: 'restyle', label: 't = ' + s, args: [attr, [0, 1, 2, 3]] };
        });

        Plotly.newPlot(el, frame(0), {
            title: { text: '<b>The Geometry of the EM Step</b><br><sup>Green: l(θ) · Orange: tangent lower bound (tight at θ⁽ᵗ⁾) · Maximize the bound → monotonic climb</sup>', x: 0.5 },
            xaxis: { title: 'θ', range: [-0.2, 6.5], gridcolor: 'lightgray' },
            yaxis: { title: 'log-likelihood', range: [-1.5, 3.2], gridcolor: 'lightgray' },
            width: 800,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90 },
            sliders: [{
                active: 0,
                currentvalue: { prefix: 'EM iteration: ', font: { size: 13 } },
                pad: { t: 30 },
                steps: steps
            }]
        }, { displayModeBar: false })
            .then(function (gd) {
                /* Backup driver: re-apply restyle args on slider updates. */
                gd.on('plotly_sliderupdate', function (ev) {
                    var step = gd.layout.sliders[0].steps[ev.slider.active];
                    if (step && step.args) Plotly.restyle(gd, step.args[0], step.args[1]);
                });
            });
    }

    /* ======================================================================
       Chart 2: PCA as Maximum-Variance Projection
       2D correlated data cloud; a slider rotates the projection direction u.
       Shows the projected points on the direction axis and the variance
       readout u^T Sigma u. Maximum spread occurs at the first eigenvector.
       ====================================================================== */
    function drawPcaProjection(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = (function (seed) {
            return function () {
                seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
                var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
                t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
                return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
            };
        })(17);

        /* Correlated 2D data (mean-centered). */
        var n = 80, pts = [];
        for (var i = 0; i < n; i++) {
            var a = rand() * 2 - 1, b = rand() * 2 - 1;
            pts.push([a * 2.2 + b * 0.5, a * 1.1 - b * 0.4]);
        }

        /* Empirical covariance and its principal eigenvector. */
        var sxx = 0, syy = 0, sxy = 0;
        pts.forEach(function (p) { sxx += p[0] * p[0]; syy += p[1] * p[1]; sxy += p[0] * p[1]; });
        var S = [[sxx / n, sxy / n], [sxy / n, syy / n]];
        /* Closed-form 2x2 eigenvector for the larger eigenvalue. */
        var thetaStar = 0.5 * Math.atan2(2 * S[0][1], S[0][0] - S[1][1]);

        function varianceAt(theta) {
            var u = [Math.cos(theta), Math.sin(theta)];
            return u[0] * (S[0][0] * u[0] + S[0][1] * u[1]) + u[1] * (S[1][0] * u[0] + S[1][1] * u[1]);
        }

        function frame(deg) {
            var theta = deg * Math.PI / 180;
            var u = [Math.cos(theta), Math.sin(theta)];
            var v = varianceAt(theta);

            /* Projected points: p -> (p·u) u, drawn faded along the line. */
            var projX = [], projY = [];
            pts.forEach(function (p) {
                var c = p[0] * u[0] + p[1] * u[1];
                projX.push(c * u[0]);
                projY.push(c * u[1]);
            });

            /* Direction line spanning the plot. */
            var L = 4.5;
            var lineX = [-L * u[0], L * u[0]];
            var lineY = [-L * u[1], L * u[1]];

            var traces = [
                {
                    x: pts.map(function (p) { return p[0]; }),
                    y: pts.map(function (p) { return p[1]; }),
                    mode: 'markers', type: 'scatter',
                    marker: { color: '#286b82', size: 7, opacity: 0.55, line: { color: 'black', width: 0.5 } },
                    name: 'Data (mean-centered)',
                    hovertemplate: '(%{x:.2f}, %{y:.2f})<extra></extra>'
                },
                {
                    x: projX, y: projY, mode: 'markers', type: 'scatter',
                    marker: { color: '#d16a1a', size: 6, opacity: 0.85, symbol: 'x' },
                    name: 'Projected points (xᵀu)u',
                    hovertemplate: 'projection: (%{x:.2f}, %{y:.2f})<extra></extra>'
                },
                {
                    x: lineX, y: lineY, mode: 'lines', type: 'scatter',
                    line: { color: '#7a1f5c', width: 2.5 },
                    name: 'u at ' + deg + '°',
                    hoverinfo: 'skip'
                },
                {
                    x: [u[0] * 2.6], y: [u[1] * 2.6], mode: 'markers+text', type: 'scatter',
                    marker: { color: '#7a1f5c', size: 10, symbol: 'arrow-bar-up', angleref: 'previous', angle: deg },
                    text: ['u'], textposition: 'top center', textfont: { size: 14 },
                    name: 'Direction u',
                    hoverinfo: 'skip', showlegend: false
                }
            ];
            return { traces: traces, v: v };
        }

        var steps = [];
        for (var deg = 0; deg <= 180; deg += 10) {
            (function (deg) {
                var f = frame(deg);
                var attr = { x: [], y: [], marker: [], line: [] };
                f.traces.forEach(function (tr) {
                    attr.x.push(tr.x);
                    attr.y.push(tr.y);
                    attr.marker.push(tr.marker || null);
                    attr.line.push(tr.line || null);
                });
                steps.push({
                    method: 'restyle',
                    label: deg + '°',
                    args: [attr, [0, 1, 2, 3]]
                });
            })(deg);
        }

        var f0 = frame(0);
        Plotly.newPlot(el, f0.traces, {
            title: { text: '<b>PCA: Maximum-Variance Projection</b><br><sup>Variance uᵀΣu at 0°: ' + f0.v.toFixed(2) + ' — rotate u toward ' + (thetaStar * 180 / Math.PI).toFixed(0) + '° to maximize it</sup>', x: 0.5 },
            xaxis: { title: 'x₁', range: [-4.5, 4.5], gridcolor: 'lightgray', zeroline: true, zerolinecolor: '#bbb' },
            yaxis: { title: 'x₂', range: [-3.2, 3.2], gridcolor: 'lightgray', zeroline: true, zerolinecolor: '#bbb', scaleanchor: 'x', scaleratio: 1 },
            width: 800,
            height: 600,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90 },
            sliders: [{
                active: 0,
                currentvalue: { prefix: 'direction u: ', font: { size: 13 } },
                pad: { t: 30 },
                steps: steps
            }]
        }, { displayModeBar: false })
            .then(function (gd) {
                /* Backup driver + live variance readout in the title. */
                gd.on('plotly_sliderupdate', function (ev) {
                    var active = ev.slider.active;
                    var deg = active * 10;
                    var step = gd.layout.sliders[0].steps[active];
                    if (step && step.args) Plotly.restyle(gd, step.args[0], step.args[1]);
                    var v = varianceAt(deg * Math.PI / 180);
                    Plotly.relayout(gd, {
                        'title.text': '<b>PCA: Maximum-Variance Projection</b><br><sup>u at ' + deg + '° → variance uᵀΣu = ' + v.toFixed(2) + ' (max at ' + (thetaStar * 180 / Math.PI).toFixed(0) + '°)</sup>'
                    });
                });
            });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture10Charts = function () {
        drawEmGeometry('plotly-em-geometry');
        drawPcaProjection('plotly-pca-projection');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture10Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-em-geometry')) {
        window.renderLecture10Charts();
    }
})();
