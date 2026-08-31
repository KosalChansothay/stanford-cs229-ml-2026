/* ==========================================================================
   Lecture 8 interactive Plotly charts.
   Loaded by courses/cs229/lecture-8.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-8.html).
   ========================================================================== */

(function () {
    'use strict';

    /* ======================================================================
       Chart: Computational Graph of Backpropagation
       Top row: forward pass (left -> right): x -> [W1] -> a1 -> [W2] -> a2 -> J
       Bottom row: backward pass (right -> left): dJ -> delta2 -> gamma2 ->
       delta1 -> gamma1, with branch-offs showing the rank-1 weight gradients.
       Hover any node for its formula. A slider steps through the phases.
       ====================================================================== */
    function drawBackpropGraph(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* Node definitions: id, x, y, label, color, hover text, phases [from,to]
           Phases: 0 = forward, 1 = backward, 2 = parameter grads (all shown at end). */
        var FWD = '#286b82', BWD = '#d16a1a', GRAD = '#7a1f5c', IO = '#173f5f';

        var nodes = [
            { id: 'x', x: 0.8, y: 3.4, phase: 0, color: IO, label: '<b>x</b><br>input', hover: 'Input activation a⁽⁰⁾ = x enters the forward pass.' },
            { id: 'w1', x: 2.3, y: 3.4, phase: 0, color: FWD, label: '<b>W⁽¹⁾</b>', hover: 'Forward: z⁽¹⁾ = W⁽¹⁾a⁽⁰⁾ + b⁽¹⁾ — affine layer 1.' },
            { id: 'a1', x: 3.6, y: 3.4, phase: 0, color: FWD, label: '<b>a⁽¹⁾</b><br>σ(z⁽¹⁾)', hover: 'Forward: a⁽¹⁾ = σ(z⁽¹⁾) — entrywise activation. Cached for the backward pass.' },
            { id: 'w2', x: 5.1, y: 3.4, phase: 0, color: FWD, label: '<b>W⁽²⁾</b>', hover: 'Forward: z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾ — affine layer 2.' },
            { id: 'a2', x: 6.4, y: 3.4, phase: 0, color: FWD, label: '<b>a⁽²⁾</b><br>σ(z⁽²⁾)', hover: 'Forward: a⁽²⁾ = σ(z⁽²⁾) — final hidden activation. Cached.' },
            { id: 'J', x: 7.9, y: 3.4, phase: 0, color: IO, label: '<b>J</b><br>loss', hover: 'Scalar loss J = L(a⁽ᴸ⁾, y). Backpropagation starts here.' },
            { id: 'dJ', x: 7.9, y: 1.9, phase: 1, color: BWD, label: '<b>∂J/∂a⁽²⁾</b>', hover: 'Backward seed: δ⁽ᴸ⁾ = ∇_{a⁽ᴸ⁾} L(a⁽ᴸ⁾, y).' },
            { id: 'g2', x: 6.4, y: 1.9, phase: 1, color: BWD, label: '<b>γ⁽²⁾</b><br>δ⊙σ′', hover: 'Backward through activation: γ⁽²⁾ = δ⁽²⁾ ⊙ σ′(z⁽²⁾) — Hadamard product.' },
            { id: 'wt2', x: 5.1, y: 1.9, phase: 1, color: BWD, label: '<b>(W⁽²⁾)ᵀ</b>', hover: 'Backward through affine: δ⁽¹⁾ = (W⁽²⁾)ᵀ γ⁽²⁾ — transposed weights.' },
            { id: 'g1', x: 3.6, y: 1.9, phase: 1, color: BWD, label: '<b>γ⁽¹⁾</b><br>δ⊙σ′', hover: 'Backward through activation: γ⁽¹⁾ = δ⁽¹⁾ ⊙ σ′(z⁽¹⁾).' },
            { id: 'gw1', x: 2.3, y: 0.6, phase: 2, color: GRAD, label: '<b>∂J/∂W⁽¹⁾</b><br>= γ⁽¹⁾(a⁽⁰⁾)ᵀ', hover: 'Rank-1 outer product: γ⁽¹⁾ (a⁽⁰⁾)ᵀ — the Hebbian-style weight gradient.' },
            { id: 'gw2', x: 5.1, y: 0.6, phase: 2, color: GRAD, label: '<b>∂J/∂W⁽²⁾</b><br>= γ⁽²⁾(a⁽¹⁾)ᵀ', hover: 'Rank-1 outer product: γ⁽²⁾ (a⁽¹⁾)ᵀ — needs the cached forward activation a⁽¹⁾.' }
        ];

        /* Forward edges (top row, left to right). */
        var fwdEdges = [
            ['x', 'w1'], ['w1', 'a1'], ['a1', 'w2'], ['w2', 'a2'], ['a2', 'J']
        ];
        /* Backward edges (bottom row, right to left). */
        var bwdEdges = [
            ['J', 'dJ'], ['dJ', 'g2'], ['g2', 'wt2'], ['wt2', 'g1']
        ];
        /* Branch edges: backward row -> parameter gradient nodes. */
        var branchEdges = [
            ['g1', 'gw1'], ['g2', 'gw2']
        ];
        /* Vertical connectors: forward node -> its backward counterpart. */
        var dropEdges = [
            ['J', 'dJ'], ['a2', 'g2'], ['a1', 'g1']
        ];

        var byId = {};
        nodes.forEach(function (n) { byId[n.id] = n; });

        function edgeTrace(pairs, color, dash, width) {
            var ex = [], ey = [];
            pairs.forEach(function (p) {
                var a = byId[p[0]], b = byId[p[1]];
                ex.push(a.x, b.x, null);
                ey.push(a.y, b.y, null);
            });
            return {
                x: ex, y: ey, mode: 'lines', type: 'scatter',
                line: { color: color, width: width || 2.5, dash: dash || 'solid' },
                hoverinfo: 'skip', showlegend: false
            };
        }

        var nodeTrace = {
            x: nodes.map(function (n) { return n.x; }),
            y: nodes.map(function (n) { return n.y; }),
            mode: 'markers+text', type: 'scatter',
            marker: {
                size: 44,
                color: nodes.map(function (n) { return n.color; }),
                opacity: 0.22,
                line: { color: nodes.map(function (n) { return n.color; }), width: 2.5 }
            },
            text: nodes.map(function (n) { return n.label; }),
            textposition: 'bottom center',
            textfont: { size: 11.5, color: '#26343b' },
            hovertext: nodes.map(function (n) { return n.hover; }),
            hoverinfo: 'text',
            showlegend: false
        };

        /* Row labels. */
        var rowLabels = {
            annotations: [
                { text: '<b>Forward pass →</b>', x: 0.8, y: 4.15, xref: 'x', yref: 'y', showarrow: false, font: { size: 13, color: FWD } },
                { text: '<b>← Backward pass</b>', x: 7.9, y: 1.25, xref: 'x', yref: 'y', showarrow: false, font: { size: 13, color: BWD } },
                { text: '<b>Parameter gradients (rank-1)</b>', x: 3.7, y: 0.02, xref: 'x', yref: 'y', showarrow: false, font: { size: 13, color: GRAD } }
            ]
        };

        var data = [
            edgeTrace(fwdEdges, FWD, 'solid', 3),
            edgeTrace(dropEdges, '#8a9798', 'dot', 1.5),
            edgeTrace(bwdEdges, BWD, 'solid', 3),
            edgeTrace(branchEdges, GRAD, 'dash', 2.5),
            nodeTrace
        ];

        Plotly.newPlot(el, data, {
            title: { text: '<b>Computational Graph of Backpropagation</b><br><sup>Forward (blue, →) · Backward (orange, ←) · Rank-1 weight gradients (purple, dashed) — hover any node</sup>', x: 0.5 },
            xaxis: { range: [0, 8.7], visible: false },
            yaxis: { range: [-0.3, 4.5], visible: false, scaleanchor: 'x', scaleratio: 1 },
            width: 880,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            annotations: rowLabels.annotations,
            margin: { t: 90, b: 30 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture8Charts = function () {
        drawBackpropGraph('plotly-backprop-graph');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture8Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-backprop-graph')) {
        window.renderLecture8Charts();
    }
})();
