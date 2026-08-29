/* ==========================================================================
   Lecture 9 interactive Plotly charts.
   Loaded by courses/cs229/lecture-9.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-9.html).
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

    /* ======================================================================
       Chart 1: K-Means Iteration Steps
       Real K-Means executed step-by-step; a slider advances through
       (assign, update) phases. Points recolor by assignment; centroids move.
       ====================================================================== */
    function drawKMeans(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = mulberry32(31);

        /* Three natural blobs. */
        var blobs = [
            { center: [-2.2, 1.8], spread: 0.9 },
            { center: [2.4, 2.2], spread: 0.8 },
            { center: [0.2, -1.9], spread: 0.95 }
        ];
        var pts = [];
        blobs.forEach(function (b) {
            for (var i = 0; i < 40; i++) {
                pts.push([
                    b.center[0] + (rand() - 0.5) * 2 * b.spread * 2,
                    b.center[1] + (rand() - 0.5) * 2 * b.spread * 2
                ]);
            }
        });

        /* Deliberately suboptimal but spread initialization: all in the lower
           half, so the animation shows real movement (and motivates K-Means++). */
        var k = 3;
        var centroids = [[-3.4, -3.2], [0.0, -4.2], [3.4, -3.4]];

        /* Run K-Means, recording state after each half-step. */
        var states = [{ cents: JSON.parse(JSON.stringify(centroids)), assign: null, label: 'Initial centroids' }];
        var assign = null;
        for (var iter = 0; iter < 6; iter++) {
            /* Assignment step. */
            assign = pts.map(function (p) {
                var best = 0, bestD = Infinity;
                for (var j = 0; j < k; j++) {
                    var dx = p[0] - centroids[j][0], dy = p[1] - centroids[j][1];
                    var d = dx * dx + dy * dy;
                    if (d < bestD) { bestD = d; best = j; }
                }
                return best;
            });
            states.push({ cents: JSON.parse(JSON.stringify(centroids)), assign: assign.slice(), label: 'Iteration ' + (iter + 1) + ': assign' });

            /* Update step. */
            for (var j2 = 0; j2 < k; j2++) {
                var sx = 0, sy = 0, cnt = 0;
                for (var i2 = 0; i2 < pts.length; i2++) {
                    if (assign[i2] === j2) { sx += pts[i2][0]; sy += pts[i2][1]; cnt++; }
                }
                if (cnt > 0) { centroids[j2] = [sx / cnt, sy / cnt]; }
            }
            states.push({ cents: JSON.parse(JSON.stringify(centroids)), assign: assign.slice(), label: 'Iteration ' + (iter + 1) + ': update' });
        }

        var palette = ['#d62728', '#1f77b4', '#2ca02c'];

        function frame(s) {
            var st = states[s];
            var traces = [];
            for (var j = 0; j < k; j++) {
                var members = st.assign
                    ? pts.filter(function (_, i) { return st.assign[i] === j; })
                    : [];
                traces.push({
                    x: members.map(function (p) { return p[0]; }),
                    y: members.map(function (p) { return p[1]; }),
                    mode: 'markers', type: 'scatter',
                    marker: { color: palette[j], size: 8, opacity: 0.75, line: { color: 'black', width: 0.5 } },
                    name: 'Cluster ' + (j + 1),
                    hovertemplate: '(%{x:.2f}, %{y:.2f})<extra>Cluster ' + (j + 1) + '</extra>'
                });
            }
            traces.push({
                x: st.cents.map(function (c) { return c[0]; }),
                y: st.cents.map(function (c) { return c[1]; }),
                mode: 'markers+text', type: 'scatter',
                marker: { color: palette, size: 17, symbol: 'x', line: { color: 'black', width: 2 } },
                text: ['μ₁', 'μ₂', 'μ₃'], textposition: 'top center', textfont: { size: 14 },
                name: 'Centroids',
                hovertemplate: 'centroid: (%{x:.2f}, %{y:.2f})<extra></extra>'
            });
            return { data: traces };
        }

        var steps = states.map(function (st, s) {
            /* 'restyle' method: args = [attrObject, traceIndices].
               attrObject values must be arrays-of-arrays (one entry per trace). */
            var f = frame(s);
            var attr = { x: [], y: [], marker: [], text: [], line: [] };
            f.data.forEach(function (tr) {
                attr.x.push(tr.x);
                attr.y.push(tr.y);
                attr.marker.push(tr.marker);
                attr.text.push(tr.text || null);
                attr.line.push(tr.line || null);
            });
            return {
                method: 'restyle',
                label: s === 0 ? 'init' : String(s),
                args: [attr, [0, 1, 2, 3]]
            };
        });

        Plotly.newPlot(el, frame(0).data, {
            title: { text: '<b>K-Means: Step-by-Step</b><br><sup>' + states[0].label + ' — drag the slider to assign points and move centroids</sup>', x: 0.5 },
            xaxis: { title: 'x₁', range: [-5.5, 5.5], gridcolor: 'lightgray', zeroline: true, zerolinecolor: '#ccc' },
            yaxis: { title: 'x₂', range: [-5.5, 5.5], gridcolor: 'lightgray', zeroline: true, zerolinecolor: '#ccc', scaleanchor: 'x', scaleratio: 1 },
            width: 760,
            height: 620,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            plot_bgcolor: 'white',
            margin: { t: 80 },
            sliders: [{
                active: 0,
                currentvalue: { prefix: 'step: ', font: { size: 13 } },
                pad: { t: 30 },
                steps: steps
            }]
        }, { displayModeBar: false })
            .then(function (gd) {
                /* Drive the data update ourselves via Plotly.react. */
                gd.on('plotly_sliderupdate', function (ev) {
                    var s = ev.slider.active;
                    Plotly.react(gd, frame(s).data, gd.layout);
                });
            });
    }

    /* ======================================================================
       Chart 2: The EM Lower Bound (Jensen)
       A 1-D non-convex log-likelihood and tangent lower bounds; a slider
       advances EM iterations, each maximizing the bound then re-tying it.
       ====================================================================== */
    function drawEmBound(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* True log-likelihood: smooth double-hump (non-convex). */
        function ell(t) {
            return 2.2 * Math.exp(-Math.pow(t - 1.2, 2) / 0.9)
                + 1.6 * Math.exp(-Math.pow(t - 4.2, 2) / 1.8)
                + 0.15 * t - 1.2;
        }
        /* Derivative (for tangent slopes). */
        function dell(t) {
            var e1 = 2.2 * Math.exp(-Math.pow(t - 1.2, 2) / 0.9);
            var e2 = 1.6 * Math.exp(-Math.pow(t - 4.2, 2) / 1.8);
            return e1 * (-2 * (t - 1.2) / 0.9) + e2 * (-2 * (t - 4.2) / 1.8) + 0.15;
        }

        /* EM trajectory: start left of the first hump; each step jumps to the
           maximizer of the tangent lower bound (simulated by gradient ascent
           on the bound from the current point). */
        var thetas = [0.2];
        for (var it = 0; it < 5; it++) {
            var t = thetas[thetas.length - 1];
            /* Maximize L_t(u) = ell(t) + dell(t)*(u - t) over u: since the bound
               is linear, EM on a linear bound is simulated by a damped ascent
               toward where the bound would touch the next hump. */
            var u = t;
            for (var s = 0; s < 30; s++) {
                var slope = dell(u);
                u = u + 0.35 * slope;      // ascent on ell itself (bound max ties to ell)
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
                name: 'Lower bound L_t(θ) (tangent at θ = ' + t.toFixed(2) + ')',
                hovertemplate: 'bound: %{y:.2f}<extra></extra>'
            };
        }

        function frame(step) {
            var t = thetas[step];
            var traces = [
                {
                    x: xs, y: ellY, mode: 'lines', type: 'scatter',
                    line: { color: '#2ca02c', width: 3 },
                    name: 'True log-likelihood ℓ(θ)',
                    hovertemplate: 'θ: %{x:.2f}<br>ℓ(θ): %{y:.2f}<extra></extra>'
                },
                boundTrace(t),
                {
                    x: [t], y: [ell(t)], mode: 'markers', type: 'scatter',
                    marker: { color: '#7a1f5c', size: 12, symbol: 'circle' },
                    name: 'θ⁽ᵗ⁾ = ' + t.toFixed(2),
                    hovertemplate: 'current θ: %{x:.2f}<br>ℓ: %{y:.2f}<extra>tangent point (bound is tight)</extra>'
                },
                {
                    /* Trajectory: always present (empty at step 0) so trace
                       indices stay stable for slider restyle updates. */
                    x: step > 0 ? thetas.slice(0, step + 1) : [],
                    y: step > 0 ? thetas.slice(0, step + 1).map(ell) : [],
                    mode: 'lines+markers', type: 'scatter',
                    line: { color: '#7a1f5c', width: 1.5, dash: 'dot' },
                    marker: { size: 6, symbol: 'circle-open' },
                    name: 'EM trajectory',
                    hoverinfo: 'skip'
                }
            ];
            return traces;
        }

        var steps = thetas.map(function (_, s) {
            /* 'restyle' method: args = [attrObject, traceIndices]. */
            var f = frame(s);
            var attr = { x: [], y: [], line: [], marker: [] };
            f.forEach(function (tr) {
                attr.x.push(tr.x);
                attr.y.push(tr.y);
                attr.line.push(tr.line || null);
                attr.marker.push(tr.marker || null);
            });
            return {
                method: 'restyle',
                label: 't = ' + s,
                args: [attr, [0, 1, 2, 3]]
            };
        });

        Plotly.newPlot(el, frame(0), {
            title: { text: '<b>The EM Lower Bound via Jensen\'s Inequality</b><br><sup>Green: true ℓ(θ) · Red: tangent bound L_t (tight at θ⁽ᵗ⁾) · Maximize the bound → climb the surface</sup>', x: 0.5 },
            xaxis: { title: 'θ', range: [-0.2, 6.5], gridcolor: 'lightgray' },
            yaxis: { title: 'log-likelihood', range: [-1.5, 3.2], gridcolor: 'lightgray' },
            width: 800,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            plot_bgcolor: 'white',
            margin: { t: 80 },
            sliders: [{
                active: 0,
                currentvalue: { prefix: 'EM iteration: ', font: { size: 13 } },
                pad: { t: 30 },
                steps: steps
            }]
        }, { displayModeBar: false })
            .then(function (gd) {
                /* Backup driver: re-apply the step's restyle args on slider
                   updates (guarantees the data moves in every Plotly build). */
                gd.on('plotly_sliderupdate', function (ev) {
                    var step = gd.layout.sliders[0].steps[ev.slider.active];
                    if (step && step.args) Plotly.restyle(gd, step.args[0], step.args[1]);
                });
            });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture9Charts = function () {
        drawKMeans('plotly-kmeans-steps');
        drawEmBound('plotly-em-bound');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture9Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-kmeans-steps')) {
        window.renderLecture9Charts();
    }
})();
