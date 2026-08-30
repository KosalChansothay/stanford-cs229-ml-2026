/* ==========================================================================
   Lecture 13 interactive Plotly charts — Contrastive Learning & RAG.
   Chart 1: Embedding space with positive pairs pulled together and
            negative pairs pushed apart (SimCLR force balance).
   Chart 2: NT-Xent loss vs positive similarity for several negative-sum
            values, showing the monotone decreasing gradient.
   Loaded by courses/cs229/lecture-13.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in lecture-13.html).
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

    function gaussian(rand) {
        var u = Math.max(rand(), 1e-12), v = rand();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    /* ======================================================================
       Chart 1: 2-D embedding space. Three semantic clusters (cats, dogs,
       planes). Positive pairs (two augmentations of the same image) get an
       attractive arrow; cross-class pairs get a repulsive arrow.
       ====================================================================== */
    function drawEmbeddingSpace(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rand = mulberry32(13);
        var navy = '#173f5f', red = '#c0392b', green = '#2e7d4f';

        /* Cluster centers: cats, dogs, airplanes. */
        var clusters = [
            { name: 'Cat images', cx: -2.2, cy: 1.4, color: '#286b82' },
            { name: 'Dog images', cx: 2.0, cy: 1.8, color: '#8a6d3b' },
            { name: 'Airplanes', cx: 0.2, cy: -2.0, color: '#7a4b8f' }
        ];

        var traces = [];
        clusters.forEach(function (c) {
            var xs = [], ys = [];
            for (var i = 0; i < 12; i++) {
                xs.push(c.cx + gaussian(rand) * 0.55);
                ys.push(c.cy + gaussian(rand) * 0.55);
            }
            traces.push({
                x: xs, y: ys, mode: 'markers', name: c.name,
                marker: { color: c.color, size: 8, opacity: 0.75 },
                hovertemplate: c.name + '<extra></extra>'
            });
        });

        /* Positive pair: two augmentations of the same cat — pulled together. */
        var p1 = [-2.6, 1.1], p2 = [-1.9, 1.7];
        traces.push({
            x: [p1[0], p2[0]], y: [p1[1], p2[1]], mode: 'markers',
            marker: { color: red, size: 12, symbol: 'star' },
            name: 'Positive pair (same image, 2 augmentations)',
            hovertemplate: 'augmentation of the same cat<extra></extra>'
        });

        /* Negative pair: cat vs airplane — pushed apart. */
        var n1 = [-1.6, 0.6], n2 = [-0.4, -1.2];

        function arrow(x0, y0, x1, y1, color, name) {
            traces.push({
                x: [x0, x1], y: [y0, y1], mode: 'lines',
                line: { color: color, width: 2.5 },
                name: name, showlegend: false,
                hoverinfo: 'skip'
            });
        }

        /* Attractive arrows pointing inward for the positive pair. */
        arrow(p1[0] + 0.18, p1[1] + 0.18, p1[0] + 0.42, p1[1] + 0.42, green);
        arrow(p2[0] - 0.18, p2[1] - 0.18, p2[0] - 0.42, p2[1] - 0.42, green);
        /* Repulsive arrows pointing outward for the negative pair. */
        arrow((n1[0] + n2[0]) / 2, (n1[1] + n2[1]) / 2, n1[0] - 0.35, n1[1] + 0.35, red);
        arrow((n1[0] + n2[0]) / 2, (n1[1] + n2[1]) / 2, n2[0] + 0.35, n2[1] - 0.35, red);

        var annotations = [
            {
                x: -2.25, y: 1.4, xref: 'x', yref: 'y', text: 'attract (maximize A)',
                showarrow: false, font: { color: green, size: 11 }, yshift: 34
            },
            {
                x: -1.0, y: -0.3, xref: 'x', yref: 'y', text: 'repel (minimize B)',
                showarrow: false, font: { color: red, size: 11 }, yshift: -18
            }
        ];

        var layout = {
            title: { text: '<b>SimCLR Force Balance in Embedding Space</b>', x: 0.5, font: { size: 16 } },
            xaxis: { title: 'Embedding dim 1', range: [-4.5, 4.5], gridcolor: 'lightgray', zeroline: true, zerolinecolor: '#d9ddd8' },
            yaxis: { title: 'Embedding dim 2', range: [-4, 4], gridcolor: 'lightgray', scaleanchor: 'x', scaleratio: 1 },
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.01, xanchor: 'left', yanchor: 'bottom', bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'lightgray', borderwidth: 1, font: { size: 10 } },
            annotations: annotations,
            plot_bgcolor: 'white',
            margin: { t: 55 }
        };

        Plotly.newPlot(el, traces, layout, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: NT-Xent loss L = -log( A / (A + B) ) with A = exp(S+/tau),
       B = sum of negative exp terms. Shows loss decreasing in A for fixed
       B (attract) and increasing in B for fixed A (repel).
       ====================================================================== */
    function drawContrastiveLoss(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var tau = 0.5;
        var sPos = [], lossB1 = [], lossB5 = [], lossB20 = [];
        for (var s = -0.9; s <= 0.95; s += 0.02) {
            var A = Math.exp(s / tau);
            sPos.push(s);
            [1, 5, 20].forEach(function (Bn, k) {
                var B = Bn * Math.exp(0 / tau); /* negatives at similarity 0 */
                var L = -Math.log(A / (A + B));
                if (k === 0) lossB1.push(L);
                else if (k === 1) lossB5.push(L);
                else lossB20.push(L);
            });
        }

        var data = [
            { x: sPos, y: lossB1, mode: 'lines', name: '1 negative (B small)', line: { color: '#2e7d4f', width: 2.5 } },
            { x: sPos, y: lossB5, mode: 'lines', name: '5 negatives', line: { color: '#286b82', width: 2.5 } },
            { x: sPos, y: lossB20, mode: 'lines', name: '20 negatives (B large)', line: { color: '#c0392b', width: 2.5 } }
        ];

        var layout = {
            title: { text: '<b>NT-Xent Loss vs Positive Similarity S₊ (τ = 0.5)</b>', x: 0.5, font: { size: 16 } },
            xaxis: { title: 'Positive-pair cosine similarity S₊', gridcolor: 'lightgray' },
            yaxis: { title: 'Contrastive loss ℒ', gridcolor: 'lightgray' },
            height: 460,
            responsive: true,
            legend: { x: 0.65, y: 0.98, bgcolor: 'rgba(255,255,255,0.75)', bordercolor: 'lightgray', borderwidth: 1, font: { size: 11 } },
            plot_bgcolor: 'white',
            margin: { t: 55 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 3: RAG system architecture flow diagram.
       ====================================================================== */
    function drawRagArchitecture(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var navy = '#173f5f', red = '#c0392b', green = '#2e7d4f', gray = '#8a9798';

        var data = [{
            x: [0, 12], y: [0, 7],
            mode: 'markers', marker: { opacity: 0 }, hoverinfo: 'skip', showlegend: false
        }];

        function box(x, y, w, h, color) {
            return {
                type: 'rect', xref: 'x', yref: 'y',
                x0: x, y0: y, x1: x + w, y1: y + h,
                line: { color: color, width: 2 }, fillcolor: 'rgba(255,253,249,0.95)'
            };
        }
        function ann(x, y, text, color, size) {
            return {
                xref: 'x', yref: 'y', x: x, y: y, text: text, showarrow: false,
                font: { color: color || '#26343b', size: size || 12 },
                xanchor: 'center', yanchor: 'middle'
            };
        }
        function arrow(x0, y0, x1, y1, color) {
            return {
                xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                x: x1, y: y1, ax: x0, ay: y0,
                showarrow: true, arrowhead: 2, arrowsize: 1.1,
                arrowwidth: 1.8, arrowcolor: color || '#637179'
            };
        }

        var shapes = [
            box(0.3, 3.1, 1.9, 1.0, gray),      /* User query */
            box(3.0, 3.1, 2.0, 1.0, navy),      /* Embed */
            box(5.8, 3.1, 2.4, 1.0, red),       /* Vector DB */
            box(5.8, 0.6, 2.4, 1.0, gray),      /* Top-K docs */
            box(9.0, 3.1, 2.6, 1.0, green)      /* LLM */
        ];

        var arrows = [
            arrow(2.2, 3.6, 3.0, 3.6),          /* query -> embed */
            arrow(5.0, 3.6, 5.8, 3.6),          /* embed -> DB */
            arrow(7.0, 3.1, 7.0, 1.6),          /* DB -> docs */
            arrow(7.0, 1.6, 9.6, 3.1, green),   /* docs -> LLM */
            arrow(8.2, 3.6, 9.0, 3.6)           /* query -> LLM */
        ];

        var annotations = [
            ann(1.25, 3.6, '1. User query', '#26343b', 11),
            ann(4.0, 3.6, '2. Embed φ(q)', '#ffffff', 11),
            ann(7.0, 3.6, '3. Vector DB', '#ffffff', 11),
            ann(7.0, 1.1, '4. Top-K docs', '#26343b', 11),
            ann(10.3, 3.6, '5. LLM generates', '#ffffff', 11),
            ann(10.3, 2.6, 'grounded answer', green, 10),
            ann(6.0, 5.6, 'Offline: corpus → φ(d) → indexed chunks', gray, 11)
        ];

        var layout = {
            title: { text: '<b>Retrieval-Augmented Generation (RAG) Pipeline</b>', x: 0.5, font: { size: 16 } },
            xaxis: { visible: false, range: [0, 12] },
            yaxis: { visible: false, range: [0, 6.4] },
            height: 440,
            responsive: true,
            shapes: shapes.concat(arrows),
            annotations: annotations,
            plot_bgcolor: 'rgba(242,240,235,0.4)',
            margin: { t: 55, l: 10, r: 10, b: 10 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderLecture13Charts = function () {
        drawEmbeddingSpace('plotly-13-embedding-space');
        drawContrastiveLoss('plotly-13-contrastive-loss');
        drawRagArchitecture('plotly-13-rag-architecture');
    };

    document.addEventListener('markdown:rendered', window.renderLecture13Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-13-embedding-space')) {
        window.renderLecture13Charts();
    }
})();
