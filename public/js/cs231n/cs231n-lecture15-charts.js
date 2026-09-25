/* ==========================================================================
   CS231N Lecture 15 interactive Plotly charts.
   Implements the Visualization Blueprint from the lecture notes.
   Loaded by courses/cs231n/lecture-15.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in the lecture page).
   ========================================================================== */

(function () {
    'use strict';

    /* Voxel memory O(N^3) vs NeRF/GS memory scaling with resolution. */
    function drawBlueprint(divId) {
        var el = document.getElementById(divId);
        if (!el) return;
        var Ns = [32, 64, 128, 256, 512, 1024];
        var voxel = Ns.map(function (N) { return N * N * N * 4 / 1e9; });
        var pc = Ns.map(function (N) { return N * N * 24 / 1e9; });
        var nerf = Ns.map(function (N) { return 0.05; });
        var data = [
            { x: Ns, y: voxel, mode: 'lines+markers', name: 'Voxel grid O(N^3)', line: { color: '#ff7b72', width: 3 } },
            { x: Ns, y: pc, mode: 'lines+markers', name: 'Point cloud O(N^2)', line: { color: '#79c0ff', width: 3 } },
            { x: Ns, y: nerf, mode: 'lines+markers', name: 'NeRF / 3DGS (network params)', line: { color: '#7ee787', width: 3 } }
        ];
        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: { text: '<b>3D Representation Memory vs Resolution</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { type: 'log', title: 'Resolution N (log)', tickmode: 'array', tickvals: Ns, ticktext: Ns.map(String), gridcolor: '#e5e5e5' },
            yaxis: { type: 'log', title: 'Memory (GB, log)', tickmode: 'array', tickvals: [0.01, 0.1, 1, 10, 100, 1000, 4000], ticktext: ['0.01', '0.1', '1', '10', '100', '1000', '4000'], gridcolor: '#e5e5e5' },
            height: 500, responsive: true,
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };
        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }
    window.renderCs231nLecture15Charts = function () {
        drawBlueprint('plotly-cs231n-15-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture15Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-15-blueprint')) {
        window.renderCs231nLecture15Charts();
    }
})();
