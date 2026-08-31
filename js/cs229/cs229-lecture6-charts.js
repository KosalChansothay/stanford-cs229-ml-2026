/* ==========================================================================
   Lecture 6 interactive Plotly charts.
   Loaded by courses/cs229/lecture-6.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-6.html).
   ========================================================================== */

(function () {
    'use strict';

    /* ======================================================================
       Chart 1: The Bias-Variance Trade-Off
       Model complexity (x) vs error (y). Bias^2 falls like exp(-c*d),
       variance rises like c*d, noise is constant; test error = sum (U-shape).
       A vertical marker tracks the sweet spot; hover shows exact values.
       ====================================================================== */
    function drawBiasVariance(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var dMin = 0.2, dMax = 10, n = 200;
        var noise = 0.35;
        var ds = [], bias = [], variance = [], testErr = [], trainErr = [];
        for (var i = 0; i < n; i++) {
            var d = dMin + (dMax - dMin) * i / (n - 1);
            var b = 3.2 * Math.exp(-0.55 * d);          // bias^2: falls
            var v = 0.028 * Math.exp(0.62 * d);          // variance: rises
            ds.push(d);
            bias.push(b);
            variance.push(v);
            testErr.push(b + v + noise);
            /* Training error: monotone decreasing to ~0 (slightly below test). */
            trainErr.push(Math.max(0.01, 2.4 * Math.exp(-0.5 * d)));
        }

        /* Sweet spot: minimum of test error. */
        var minIdx = 0;
        for (var k = 1; k < n; k++) if (testErr[k] < testErr[minIdx]) minIdx = k;
        var dStar = ds[minIdx];

        var data = [
            {
                x: ds, y: trainErr, mode: 'lines', type: 'scatter',
                line: { color: '#2ca02c', width: 2.5, dash: 'dash' },
                name: 'Training error (→ 0)',
                hovertemplate: 'complexity: %{x:.1f}<br>train error: %{y:.3f}<extra></extra>'
            },
            {
                x: ds, y: bias, mode: 'lines', type: 'scatter',
                line: { color: '#d62728', width: 2.5 },
                name: 'Bias² (underfitting)',
                hovertemplate: 'complexity: %{x:.1f}<br>bias²: %{y:.3f}<extra></extra>'
            },
            {
                x: ds, y: variance, mode: 'lines', type: 'scatter',
                line: { color: '#1f77b4', width: 2.5 },
                name: 'Variance (overfitting)',
                hovertemplate: 'complexity: %{x:.1f}<br>variance: %{y:.3f}<extra></extra>'
            },
            {
                x: ds, y: ds.map(function () { return noise; }), mode: 'lines', type: 'scatter',
                line: { color: '#8a9798', width: 1.5, dash: 'dot' },
                name: 'Irreducible noise σ²',
                hovertemplate: 'σ² = ' + noise + '<extra></extra>'
            },
            {
                x: ds, y: testErr, mode: 'lines', type: 'scatter',
                line: { color: '#173f5f', width: 3.5 },
                name: 'Expected test error (U-shape)',
                hovertemplate: 'complexity: %{x:.1f}<br>test error: %{y:.3f}<extra></extra>'
            },
            {
                x: [dStar, dStar], y: [0, testErr[minIdx]], mode: 'lines', type: 'scatter',
                line: { color: '#7a1f5c', width: 1.5, dash: 'dot' },
                name: 'Sweet spot',
                hoverinfo: 'skip', showlegend: false
            },
            {
                x: [dStar], y: [testErr[minIdx]], mode: 'markers+text', type: 'scatter',
                marker: { color: '#7a1f5c', size: 10, symbol: 'star' },
                text: ['sweet spot'], textposition: 'top right',
                name: 'Sweet spot',
                hovertemplate: 'complexity: ' + dStar.toFixed(1) + '<br>min test error: ' + testErr[minIdx].toFixed(3) + '<extra></extra>'
            }
        ];

        Plotly.newPlot(el, data, {
            title: { text: '<b>The Bias-Variance Trade-Off</b><br><sup>Test error = Bias² + Variance + σ² — the U-curve and its sweet spot</sup>', x: 0.5 },
            xaxis: { title: 'Model complexity (e.g., polynomial degree / parameters)', gridcolor: 'lightgray' },
            yaxis: { title: 'Error', gridcolor: 'lightgray', range: [0, 4] },
            width: 800,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.99, y: 0.99, xanchor: 'right', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: Double Descent
       Test error: classical U-curve, spike at the interpolation threshold
       (d = n), then a second descent in the overparameterized regime.
       A slider moves the sample size n and reshapes the curve.
       ====================================================================== */
    function drawDoubleDescent(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        function curve(nSamples) {
            var ds = [], test = [], train = [];
            var n = 240;
            for (var i = 0; i < n; i++) {
                var d = 0.15 + 9.85 * i / (n - 1);          // d/n ratio axis, 0.15..10
                var r = d;                                   // d relative to n = 1
                var bias = 2.6 * Math.exp(-0.5 * r);
                /* Variance grows up to the threshold, then plateaus: SGD's
                   implicit regularization (min-norm solutions) tames it. */
                var variance = 0.02 * Math.exp(0.55 * Math.min(r, 1.6));
                /* Interpolation spike near d = n: Lorentzian bump. */
                var spike = 2.2 / (1 + Math.pow((r - 1) / 0.09, 2));
                var modern = 0;                              // second-descent gain
                if (r > 1.15) modern = -0.9 * (1 - Math.exp(-(r - 1.15) / 1.6));
                var te = Math.max(0.05, bias + variance + 0.3 + spike + modern);
                var tr = r < 1 ? Math.max(0.005, 1.8 * Math.exp(-1.1 * r)) : 0.004;
                ds.push(r);
                test.push(te);
                train.push(tr);
            }
            return { ds: ds, test: test, train: train, n: nSamples };
        }

        function frame(nSamples) { return { data: buildTraces(curve(nSamples)) }; }

        function buildTraces(c) {
            return [
                {
                    x: c.ds, y: c.train, mode: 'lines', type: 'scatter',
                    line: { color: '#2ca02c', width: 2, dash: 'dash' },
                    name: 'Training error',
                    hovertemplate: 'd/n: %{x:.2f}<br>train: %{y:.3f}<extra></extra>'
                },
                {
                    x: c.ds, y: c.test, mode: 'lines', type: 'scatter',
                    line: { color: '#173f5f', width: 3.5 },
                    name: 'Test error',
                    hovertemplate: 'd/n: %{x:.2f}<br>test: %{y:.3f}<extra></extra>'
                },
                {
                    x: [1, 1], y: [0, 4], mode: 'lines', type: 'scatter',
                    line: { color: '#d62728', width: 1.5, dash: 'dot' },
                    name: 'Interpolation threshold (d = n)',
                    hoverinfo: 'skip'
                }
            ];
        }

        var c0 = curve(60);
        var steps = [];
        [20, 40, 60, 100, 200].forEach(function (nS) {
            steps.push({
                method: 'update',
                label: 'n = ' + nS,
                args: [buildTraces(curve(nS))]
            });
        });

        Plotly.newPlot(el, buildTraces(c0), {
            title: { text: '<b>Double Descent</b><br><sup>Classical U-curve → spike at d = n → second descent in the overparameterized regime</sup>', x: 0.5 },
            xaxis: { title: 'Model capacity ratio  d / n', gridcolor: 'lightgray' },
            yaxis: { title: 'Error', gridcolor: 'lightgray', range: [0, 4] },
            width: 800,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.99, y: 0.99, xanchor: 'right', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90 },
            sliders: [{
                active: 2,
                currentvalue: { prefix: 'dataset size: ', font: { size: 13 } },
                pad: { t: 30 },
                steps: steps
            }]
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture6Charts = function () {
        drawBiasVariance('plotly-bias-variance');
        drawDoubleDescent('plotly-double-descent');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture6Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-bias-variance')) {
        window.renderLecture6Charts();
    }
})();
