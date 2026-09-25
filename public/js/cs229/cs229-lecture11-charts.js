/* ==========================================================================
   Lecture 11 interactive Plotly charts — Diffusion Models.
   Chart 1: Forward noising trajectory of a 1-D signal + reverse denoising
            with the posterior mean overlay (trajectory alignment).
   Loaded by courses/cs229/lecture-11.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in lecture-11.html).
   ========================================================================== */

(function () {
    'use strict';

    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
            var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    /* Box-Muller transform for standard normal samples. */
    function gaussian(rand) {
        var u = Math.max(rand(), 1e-12), v = rand();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    /* ======================================================================
       Chart 1: Forward noising + reverse posterior mean alignment.
       We simulate the discrete forward chain on a smooth 1-D "signal"
       x_0(t) = sin curve, then plot:
         - several forward sample paths x_t (noising),
         - the closed-form marginal mean sqrt(abar_t) * x_0,
         - the reverse posterior mean mu~_t computed analytically.
       ====================================================================== */
    function drawDiffusionTrajectory(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = mulberry32(11);
        var T = 50;

        /* Noise schedule: linear beta from 1e-4 to 2e-2 (DDPM-style). */
        var betas = [], alphas = [], abars = [];
        var abar = 1;
        for (var t = 1; t <= T; t++) {
            var beta = 1e-4 + (2e-2 - 1e-4) * (t - 1) / (T - 1);
            betas.push(beta);
            var alpha = 1 - beta;
            abar *= alpha;
            alphas.push(alpha);
            abars.push(abar);
        }

        /* Clean 1-D signal over a spatial grid (the "image" axis). */
        var nPts = 120;
        var xs = [], x0 = [];
        for (var i = 0; i < nPts; i++) {
            var x = i / (nPts - 1);
            xs.push(x);
            x0.push(Math.sin(2 * Math.PI * x) * 0.8 + 0.3 * Math.sin(6 * Math.PI * x));
        }

        /* Forward sample paths at selected timesteps: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps */
        var snapTs = [1, 10, 25, 40, 50];
        var pathTraces = [];
        var palette = ['rgba(23,63,95,0.35)', 'rgba(40,107,130,0.4)', 'rgba(138,151,152,0.5)', 'rgba(200,120,60,0.45)', 'rgba(160,60,60,0.5)'];
        snapTs.forEach(function (ts, k) {
            var eps = xs.map(function () { return gaussian(rand); });
            var s = Math.sqrt(abars[ts - 1]), r = Math.sqrt(1 - abars[ts - 1]);
            pathTraces.push({
                x: xs,
                y: xs.map(function (_, i) { return s * x0[i] + r * eps[i]; }),
                mode: 'lines',
                line: { color: palette[k], width: 1.4 },
                name: 'x_' + ts + ' (noisy sample)',
                hovertemplate: 't=' + ts + ', y=%{y:.3f}<extra>x_' + ts + '</extra>'
            });
        });

        /* Closed-form marginal mean: E[x_t | x_0] = sqrt(abar_t) x_0. */
        var meanTrace = {
            x: xs,
            y: xs.map(function (_, i) { return Math.sqrt(abars[T - 1]) * x0[i]; }),
            mode: 'lines',
            line: { color: '#173f5f', width: 2.5, dash: 'dash' },
            name: 'E[x_T|x_0] ≈ 0 (signal destroyed)'
        };

        /* Reverse posterior mean mu~_t(x_t, x_0) at t = 25, computed analytically:
           mu~ = [sqrt(a_t)(1-abar_{t-1}) x_t + sqrt(abar_{t-1}) beta_t x_0] / (1 - abar_t) */
        var tRev = 25;
        var a = alphas[tRev - 1], ab1 = abars[tRev - 2], ab = abars[tRev - 1], b = betas[tRev - 1];
        var epsRev = xs.map(function () { return gaussian(rand); });
        var xt = xs.map(function (_, i) { return Math.sqrt(ab) * x0[i] + Math.sqrt(1 - ab) * epsRev[i]; });
        var postMean = xt.map(function (v, i) {
            return (Math.sqrt(a) * (1 - ab1) * v + Math.sqrt(ab1) * b * x0[i]) / (1 - ab);
        });
        var postTrace = {
            x: xs,
            y: postMean,
            mode: 'lines',
            line: { color: '#c0392b', width: 3 },
            name: 'μ̃₂₅(x₂₅, x₀) — posterior mean (target of μ_θ)'
        };
        var noisyTrace = {
            x: xs,
            y: xt,
            mode: 'lines',
            line: { color: 'rgba(192,57,43,0.35)', width: 1.2 },
            name: 'x₂₅ (noisy input)'
        };

        var data = pathTraces.concat([meanTrace, noisyTrace, postTrace]);

        var layout = {
            title: { text: '<b>Forward Noising vs. Reverse Posterior Mean</b>', x: 0.5, y: 0.98, font: { size: 16 } },
            xaxis: { title: 'Signal coordinate (flattened image axis)', gridcolor: 'lightgray' },
            yaxis: { title: 'Value', gridcolor: 'lightgray' },
            height: 520,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.75)', bordercolor: 'lightgray', borderwidth: 1, font: { size: 10 } },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: Variance schedule — why the sqrt(1-beta) scaling keeps the
       variance bounded (convex interpolation toward identity).
       ====================================================================== */
    function drawVarianceSchedule(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var T = 200;
        var ts = [], abarBad = [], abarGood = [];
        var cumBad = 1, cumGood = 1;
        for (var t = 1; t <= T; t++) {
            var beta = 1e-4 + (2e-2 - 1e-4) * (t - 1) / (T - 1);
            /* "Bad" scheme: x_t = x_{t-1} + sqrt(beta) eps  =>  Var grows as cum + beta*t */
            cumBad += beta;
            /* Correct scheme: x_t = sqrt(1-beta) x_{t-1} + sqrt(beta) eps => convex combo */
            cumGood = cumGood * (1 - beta) + beta;
            ts.push(t);
            abarBad.push(cumBad);
            abarGood.push(cumGood);
        }

        var data = [
            {
                x: ts, y: abarBad,
                mode: 'lines', name: 'Var(x_t) without scaling (explodes)',
                line: { color: '#c0392b', width: 2.5 }
            },
            {
                x: ts, y: abarGood,
                mode: 'lines', name: 'Var(x_t) with √(1−β) scaling → 1',
                line: { color: '#173f5f', width: 2.5 }
            },
            {
                x: ts, y: ts.map(function () { return 1; }),
                mode: 'lines', name: 'Identity covariance (N(0, I))',
                line: { color: '#8a9798', width: 1.5, dash: 'dot' },
                hoverinfo: 'skip'
            }
        ];

        var layout = {
            title: { text: '<b>Why Scale by √(1−β)? Variance Stability</b>', x: 0.5, y: 0.98, font: { size: 16 } },
            xaxis: { title: 'Timestep t', gridcolor: 'lightgray' },
            yaxis: { title: 'Variance of x_t', gridcolor: 'lightgray' },
            height: 460,
            responsive: true,
            legend: { x: 0.35, y: 0.98, bgcolor: 'rgba(255,255,255,0.75)', bordercolor: 'lightgray', borderwidth: 1, font: { size: 11 } },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderLecture11Charts = function () {
        drawDiffusionTrajectory('plotly-11-diffusion-trajectory');
        drawVarianceSchedule('plotly-11-variance-schedule');
    };

    document.addEventListener('markdown:rendered', window.renderLecture11Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-11-diffusion-trajectory')) {
        window.renderLecture11Charts();
    }
})();
