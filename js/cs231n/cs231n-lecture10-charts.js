/* ==========================================================================
   CS231N Lecture 10 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-10.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Spatiotemporal receptive field growth across 3D-CNN layers. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        var L = 8, layers = [], spatial = [], temporal = [];
        var s = 7, t = 1;
        for (var i = 0; i < L; i++) {
            layers.push('L' + (i + 1));
            spatial.push(s); temporal.push(t);
            s += 4; t += 2;
        }
        var data = [
            { x: layers, y: spatial, mode: 'lines+markers', name: 'Spatial extent (pixels)', line: { color: '#79c0ff', width: 3 } },
            { x: layers, y: temporal, mode: 'lines+markers', name: 'Temporal extent (frames)', yaxis: 'y2', line: { color: '#f0883e', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>3D-CNN Receptive Field Growth: Space vs Time</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Layer', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Spatial receptive field (px)', gridcolor: '#e5e5e5' },
            yaxis2: { title: 'Temporal extent (frames)', overlaying: 'y', side: 'right', gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture10Charts = function () {
        drawBlueprint('plotly-cs231n-10-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture10Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-10-blueprint')) {
        window.renderCs231nLecture10Charts();
    }
})();
