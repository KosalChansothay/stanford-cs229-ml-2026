/* ==========================================================================
   Lecture 12 interactive Plotly charts — Representation Learning & LoRA.
   Chart 1: LoRA forward pass diagram (frozen W0 + low-rank AB branch).
   Chart 2: Trainable parameter count: full fine-tuning vs LoRA vs rank.
   Loaded by courses/cs229/lecture-12.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in lecture-12.html).
   ========================================================================== */

(function () {
    'use strict';

    /* ======================================================================
       Chart 1: LoRA layer schematic drawn with Plotly shapes/annotations.
       h_in splits into the frozen W0 path and the low-rank B (d->r) then
       A (r->d) path; outputs are summed into h_out.
       ====================================================================== */
    function drawLoraDiagram(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var navy = '#173f5f', red = '#c0392b', gray = '#8a9798';

        var data = [{
            x: [0, 10], y: [0, 6],
            mode: 'markers', marker: { opacity: 0 }, hoverinfo: 'skip', showlegend: false
        }];

        function box(x, y, w, h, color, label, sub) {
            return {
                type: 'rect', xref: 'x', yref: 'y',
                x0: x, y0: y, x1: x + w, y1: y + h,
                line: { color: color, width: 2 }, fillcolor: 'rgba(255,253,249,0.9)'
            };
        }
        function ann(x, y, text, color, size) {
            return {
                xref: 'x', yref: 'y', x: x, y: y,
                text: text, showarrow: false,
                font: { color: color || '#26343b', size: size || 13 },
                xanchor: 'center', yanchor: 'middle'
            };
        }
        function arrow(x0, y0, x1, y1, color) {
            return {
                xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                x: x1, y: y1, ax: x0, ay: y0,
                showarrow: true, arrowhead: 2, arrowsize: 1,
                arrowwidth: 1.6, arrowcolor: color || '#637179'
            };
        }

        var shapes = [
            box(0.4, 2.6, 1.3, 0.8, gray),           /* h_in */
            box(3.2, 4.0, 2.2, 1.1, navy),           /* W0 frozen */
            box(3.2, 0.9, 1.0, 1.1, red),            /* B */
            box(4.6, 0.9, 1.0, 1.1, red),            /* A */
            box(7.6, 2.6, 1.3, 0.8, gray),           /* h_out */
            box(6.3, 2.6, 0.8, 0.8, gray)            /* + */
        ];

        var arrows = [
            arrow(1.7, 3.0, 3.2, 4.55),              /* h_in -> W0 */
            arrow(1.7, 3.0, 3.2, 1.45),              /* h_in -> B */
            arrow(4.2, 1.45, 4.6, 1.45, red),        /* B -> A */
            arrow(5.6, 1.45, 6.5, 2.6, red),         /* A -> + */
            arrow(5.4, 4.55, 6.5, 3.4, navy),        /* W0 -> + */
            arrow(7.1, 3.0, 7.6, 3.0)                /* + -> h_out */
        ];

        var annotations = [
            ann(1.05, 3.0, 'h_in ∈ ℝᵈ', '#26343b', 12),
            ann(4.3, 4.55, 'W₀ (frozen) ∈ ℝᵈˣᵈ', '#ffffff', 13),
            ann(3.7, 1.45, 'B ∈ ℝʳˣᵈ', '#ffffff', 11),
            ann(5.1, 1.45, 'A ∈ ℝᵈˣʳ', '#ffffff', 11),
            ann(6.7, 3.0, '+', '#26343b', 16),
            ann(8.25, 3.0, 'h_out', '#26343b', 12),
            ann(4.3, 5.6, 'frozen path', navy, 11),
            ann(4.4, 0.35, 'trainable low-rank path (rank r ≪ d)', red, 11)
        ];

        var layout = {
            title: { text: '<b>LoRA Forward Pass: h_out = W₀h_in + AB·h_in</b>', x: 0.5, y: 0.98, font: { size: 16 } },
            xaxis: { visible: false, range: [0, 10] },
            yaxis: { visible: false, range: [0, 6.2] },
            height: 420,
            responsive: true,
            shapes: shapes.concat(arrows),
            annotations: annotations,
            plot_bgcolor: 'rgba(242,240,235,0.4)',
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false, staticPlot: false });
    }

    /* ======================================================================
       Chart 2: Trainable parameters — full FT (d^2) vs LoRA (2dr) as a
       function of rank r, for d = 4096. Log scale.
       ====================================================================== */
    function drawParamComparison(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var d = 4096;
        var ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256];
        var rs = ranks.map(function (r) { return 'r=' + r; });
        var loraParams = ranks.map(function (r) { return 2 * d * r; });
        var fullParams = d * d;

        var data = [
            {
                x: rs, y: loraParams,
                type: 'bar', name: 'LoRA trainable params (2dr)',
                marker: { color: '#286b82' },
                hovertemplate: '%{x}: %{y:,} params<extra>LoRA</extra>'
            },
            {
                x: rs, y: ranks.map(function () { return fullParams; }),
                type: 'scatter', mode: 'lines', name: 'Full fine-tuning (d² = 16.78M)',
                line: { color: '#c0392b', width: 3, dash: 'dash' },
                hovertemplate: 'Full FT: %{y:,} params<extra></extra>'
            }
        ];

        var layout = {
            title: { text: '<b>Trainable Parameters: LoRA vs Full Fine-Tuning (d = 4096)</b>', x: 0.5, y: 0.98, font: { size: 16 } },
            xaxis: { title: 'LoRA rank r', gridcolor: 'lightgray' },
            yaxis: {
                title: 'Trainable Parameters',
                type: 'log',
                tickmode: 'array',
                tickvals: [10000, 100000, 1000000, 16777216],
                ticktext: ['10k', '100k', '1M', '16.8M (Full FT)'],
                gridcolor: 'lightgray'
            },
            height: 460,
            responsive: true,
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.75)', bordercolor: 'lightgray', borderwidth: 1, font: { size: 11 } },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderLecture12Charts = function () {
        drawLoraDiagram('plotly-12-lora-diagram');
        drawParamComparison('plotly-12-param-comparison');
    };

    document.addEventListener('markdown:rendered', window.renderLecture12Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-12-lora-diagram')) {
        window.renderLecture12Charts();
    }
})();
