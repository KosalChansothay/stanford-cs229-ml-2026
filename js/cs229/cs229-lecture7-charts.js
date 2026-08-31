/* ==========================================================================
   Lecture 7 interactive Plotly charts.
   Loaded by courses/cs229/lecture-7.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-7.html).
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
       Chart 1: The Gradient Approximation (SGD unbiasedness)
       A quadratic loss surface around theta; the true full-batch gradient
       is one black arrow; many per-example stochastic gradients are noisy
       blue arrows whose centroid lands on the true gradient.
       ====================================================================== */
    function drawSgdUnbiased(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = mulberry32(9);

        /* Loss surface: J(theta) = 0.5 * ||A theta - b||^2 with anisotropy.
           At theta0, the full-batch gradient is g = A^T (A theta0 - b). */
        var A = [[1.0, 0.35], [0.2, 0.8]];
        var theta0 = [1.6, 1.2];
        var residual = [0.9, -0.4];                       // A theta0 - b (fixed)
        var gTrue = [
            A[0][0] * residual[0] + A[1][0] * residual[1],
            A[0][1] * residual[0] + A[1][1] * residual[1]
        ];

        /* Per-example gradients: g_i = g_true + zero-mean noise. */
        var nSamples = 26;
        var arrows = [];
        var sumX = 0, sumY = 0;
        var scale = 0.9;
        for (var i = 0; i < nSamples; i++) {
            var nx = (rand() - 0.5) * 1.6;
            var ny = (rand() - 0.5) * 1.6;
            var gx = gTrue[0] + scale * nx;
            var gy = gTrue[1] + scale * ny;
            sumX += gx; sumY += gy;
            arrows.push({ gx: gx, gy: gy });
        }
        var centroid = [sumX / nSamples, sumY / nSamples];

        /* Contour grid of the loss around theta0. */
        var xs = [], ys = [], Z = [];
        for (var yv = -1.5; yv <= 3.5001; yv += 5 / 70) ys.push(Math.round(yv * 1000) / 1000);
        for (var xv = -0.5; xv <= 3.5001; xv += 4 / 70) xs.push(Math.round(xv * 1000) / 1000);
        ys.forEach(function (yy) {
            var row = xs.map(function (xx) {
                var dx = xx - theta0[0], dy = yy - theta0[1];
                var r1 = A[0][0] * dx + A[0][1] * dy - residual[0];
                var r2 = A[1][0] * dx + A[1][1] * dy - residual[1];
                return 0.5 * (r1 * r1 + r2 * r2);
            });
            Z.push(row);
        });

        /* Arrow annotations. */
        function arrow(x0, y0, dx, dy, color, width) {
            return {
                ax: x0, ay: y0, x: x0 + dx, y: y0 + dy,
                xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                showarrow: true, arrowhead: 2, arrowsize: 1.1,
                arrowwidth: width, arrowcolor: color
            };
        }

        var anns = arrows.map(function (a) {
            return arrow(theta0[0], theta0[1], a.gx * 0.55, a.gy * 0.55, 'rgba(31,119,180,0.45)', 1.5);
        });
        anns.push(arrow(theta0[0], theta0[1], gTrue[0] * 0.55, gTrue[1] * 0.55, '#111111', 4));

        var data = [{
            type: 'contour',
            x: xs, y: ys, z: Z,
            colorscale: 'Viridis', reversescale: true,
            opacity: 0.6, line: { width: 0 },
            contours: { coloring: 'lines' },
            colorbar: { title: { text: 'J(θ)', side: 'right' }, thickness: 14 },
            hoverinfo: 'skip', showlegend: false
        }, {
            x: [theta0[0]], y: [theta0[1]], mode: 'markers', type: 'scatter',
            marker: { color: 'purple', size: 11, symbol: 'square' },
            name: 'Current θ (all gradients evaluated here)',
            hovertemplate: 'θ = (%{x:.2f}, %{y:.2f})<extra></extra>'
        }, {
            x: [centroid[0]], y: [centroid[1]], mode: 'markers', type: 'scatter',
            marker: { color: 'rgba(31,119,180,0.9)', size: 12, symbol: 'circle-open', line: { width: 2.5 } },
            name: 'Centroid of stochastic gradients',
            hovertemplate: 'centroid: (%{x:.2f}, %{y:.2f})<br>≈ true gradient direction<extra></extra>'
        }, {
            x: [theta0[0] + gTrue[0] * 0.55], y: [theta0[1] + gTrue[1] * 0.55], mode: 'markers', type: 'scatter',
            marker: { color: '#111111', size: 9, symbol: 'x' },
            name: 'True full-batch gradient tip',
            hovertemplate: 'full-batch gradient tip: (%{x:.2f}, %{y:.2f})<extra></extra>'
        }];

        Plotly.newPlot(el, data, {
            title: { text: '<b>SGD is an Unbiased Gradient Estimator</b><br><sup>Blue arrows: per-example gradients · Black arrow: full-batch gradient · Hollow circle: their centroid ≈ black tip</sup>', x: 0.5 },
            xaxis: { title: 'θ₁', gridcolor: 'lightgray' },
            yaxis: { title: 'θ₂', gridcolor: 'lightgray', scaleanchor: 'x', scaleratio: 1 },
            width: 780,
            height: 620,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'lightgray', borderwidth: 1 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            annotations: anns,
            margin: { t: 90 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: Residual Block diagram
       Flow diagram: z splits into transform path (W1 -> sigma -> W2) and
       shortcut path; both meet at a + node. Hover each node for its role.
       ====================================================================== */
    function drawResidualBlock(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* Node layout (x, y in axis coords). */
        var nodes = [
            { id: 'z', x: 1.0, y: 2.5, label: '<b>z</b><br>input', color: '#173f5f', hover: 'Input activation entering the residual block.' },
            { id: 'w1', x: 2.6, y: 3.6, label: '<b>W⁽¹⁾z + b⁽¹⁾</b><br>weight layer 1', color: '#286b82', hover: 'First affine transform of the transform path.' },
            { id: 'sig', x: 4.2, y: 3.6, label: '<b>σ</b><br>activation', color: '#286b82', hover: 'Entrywise nonlinearity (ReLU/GELU/…).' },
            { id: 'w2', x: 5.8, y: 3.6, label: '<b>W⁽²⁾· + b⁽²⁾</b><br>weight layer 2', color: '#286b82', hover: 'Second affine transform: produces F(z).' },
            { id: 'skip', x: 4.2, y: 1.2, label: '<b>shortcut</b><br>identity', color: '#8a9798', hover: 'Skip connection: carries z unchanged around the transform path.' },
            { id: 'add', x: 7.0, y: 2.5, label: '<b>+</b>', color: '#d16a1a', hover: 'Elementwise addition: F(z) + z — the heart of the residual idea.' },
            { id: 'out', x: 8.4, y: 2.5, label: '<b>Res(z)</b><br>= F(z) + z', color: '#7a1f5c', hover: 'Output of the residual block.' }
        ];

        var nodeTrace = {
            x: nodes.map(function (n) { return n.x; }),
            y: nodes.map(function (n) { return n.y; }),
            mode: 'markers+text', type: 'scatter',
            marker: {
                size: 46,
                color: nodes.map(function (n) { return n.color; }),
                opacity: 0.25,
                line: { color: nodes.map(function (n) { return n.color; }), width: 2.5 }
            },
            text: nodes.map(function (n) { return n.label; }),
            textposition: 'bottom center',
            textfont: { size: 12, color: '#26343b' },
            hovertext: nodes.map(function (n) { return n.hover; }),
            hoverinfo: 'text',
            name: 'Nodes',
            showlegend: false
        };

        /* Edges: [from, to, dashed?]. */
        var edges = [
            ['z', 'w1', false], ['w1', 'sig', false], ['sig', 'w2', false],
            ['w2', 'add', false], ['z', 'skip', false], ['skip', 'add', true], ['add', 'out', false]
        ];
        var byId = {};
        nodes.forEach(function (n) { byId[n.id] = n; });

        var edgeX = [], edgeY = [], edgeHover = [];
        edges.forEach(function (e) {
            var a = byId[e[0]], b = byId[e[1]];
            edgeX.push(a.x, b.x, null);
            edgeY.push(a.y, b.y, null);
            edgeHover.push(null, null, null);
        });

        var edgeTrace = {
            x: edgeX, y: edgeY, mode: 'lines', type: 'scatter',
            line: { color: '#8a9798', width: 2.5, dash: 'solid' },
            hoverinfo: 'skip', showlegend: false
        };

        /* Dashed shortcut edge drawn separately for style. */
        var skipA = byId['z'], skipB = byId['skip'], skipC = byId['add'];
        var skipTrace = {
            x: [skipA.x, skipA.x + 0.6, skipB.x - 0.6, skipB.x, skipB.x + 0.6, skipC.x - 0.6, skipC.x],
            y: [skipA.y, skipA.y, skipB.y, skipB.y, skipB.y, skipC.y, skipC.y],
            mode: 'lines', type: 'scatter',
            line: { color: '#8a9798', width: 2.5, dash: 'dash' },
            hovertemplate: 'Skip connection: identity mapping<extra></extra>',
            showlegend: false, name: 'Skip path'
        };

        /* Plus-sign marker at the add node. */
        var plusTrace = {
            x: [byId['add'].x], y: [byId['add'].y], mode: 'text', type: 'scatter',
            text: ['+'], textfont: { size: 30, color: '#d16a1a' },
            hoverinfo: 'skip', showlegend: false
        };

        var data = [edgeTrace, skipTrace, nodeTrace, plusTrace];

        Plotly.newPlot(el, data, {
            title: { text: '<b>Residual Block: Res(z) = F(z) + z</b><br><sup>The shortcut carries z around the transform path — hover any node for its role</sup>', x: 0.5 },
            xaxis: { range: [0.2, 9.2], visible: false },
            yaxis: { range: [0.3, 4.5], visible: false, scaleanchor: 'x', scaleratio: 1 },
            width: 820,
            height: 420,
            hovermode: 'closest',
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: { t: 90, b: 40 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture7Charts = function () {
        drawSgdUnbiased('plotly-sgd-unbiased');
        drawResidualBlock('plotly-residual-block');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture7Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-sgd-unbiased')) {
        window.renderLecture7Charts();
    }
})();
