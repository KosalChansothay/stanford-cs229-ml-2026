/* ==========================================================================
   CS231N Lecture 16 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-16.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* CLIP batch-size effect on zero-shot accuracy + compositional failure. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        var Bs = [64, 256, 1024, 4096, 16384, 32768];
        var acc = Bs.map(function (B) { return 55 + 12 * Math.log10(B / 64) / Math.log10(512) - 1.5 * Math.random() * 0; });
        var data = [
            { x: Bs, y: acc, mode: 'lines+markers', name: 'Zero-shot top-1 (synthetic)', line: { color: '#79c0ff', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>CLIP Performance Grows with Contrastive Batch Size</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { type: 'log', title: 'Batch size (log)', tickmode: 'array', tickvals: Bs, ticktext: ['64', '256', '1K', '4K', '16K', '32K'], gridcolor: '#e5e5e5' },
            yaxis: { title: 'Zero-shot top-1 (%)', range: [50, 75], gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture16Charts = function () {
        drawBlueprint('plotly-cs231n-16-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture16Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-16-blueprint')) {
        window.renderCs231nLecture16Charts();
    }
})();
