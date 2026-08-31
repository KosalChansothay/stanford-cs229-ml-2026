/* ==========================================================================
   CS336 Lecture 14 interactive Plotly charts — Data (Preprocessing & Mixing).
   Chart 1: MinHash + LSH S-Curve Collision Probability vs Jaccard Similarity s
   Chart 2: Data Repetition: Validation Loss Degradation vs Epoch Count
   ========================================================================== */

(function () {
    'use strict';

    function drawLshScurve(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // LSH Collision Probability: P(candidate pair) = 1 - (1 - s^r)^b
        // where s = Jaccard similarity, b = bands, r = rows per band (b * r = total hash functions, e.g. 100)
        var sValues = [];
        for (var s = 0; s <= 1.0; s += 0.02) sValues.push(s);

        var configs = [
            { b: 20, r: 5, label: 'b=20, r=5 (Threshold ~0.55)', color: '#286b82' },
            { b: 10, r: 10, label: 'b=10, r=10 (Threshold ~0.79)', color: '#b84a39' },
            { b: 50, r: 2, label: 'b=50, r=2 (Threshold ~0.14)', color: '#3d8b57' }
        ];

        var traces = configs.map(function (c) {
            var probs = sValues.map(function (s) {
                return 1 - Math.pow(1 - Math.pow(s, c.r), c.b);
            });
            return {
                x: sValues,
                y: probs,
                mode: 'lines',
                name: c.label,
                line: { color: c.color, width: 3 }
            };
        });

        var layout = {
            title: { text: '<b>Locality-Sensitive Hashing (LSH): S-Curve Collision Probability P(s)</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Jaccard Similarity s (Document Overlap)', gridcolor: '#e5e5e5', range: [0, 1] },
            yaxis: { title: 'Probability of Candidate Match P(s)', gridcolor: '#e5e5e5', range: [0, 1.05] },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, traces, layout, { displayModeBar: false });
    }

    function drawEpochCaps(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var epochs = [1, 2, 3, 4, 5, 6, 7, 8];
        // Unique diverse web data: steady loss improvement
        var uniqueDataLoss = [2.45, 2.32, 2.24, 2.18, 2.13, 2.09, 2.06, 2.03];
        // Repeated data: loss plateaus and overfits after epoch 4
        var repeatedDataLoss = [2.45, 2.33, 2.27, 2.25, 2.28, 2.34, 2.42, 2.51];

        var data = [
            {
                x: epochs.map(function (e) { return 'Epoch ' + e; }),
                y: uniqueDataLoss,
                mode: 'lines+markers',
                name: 'Unique Data Mixture (Infinite Horizon)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            },
            {
                x: epochs.map(function (e) { return 'Epoch ' + e; }),
                y: repeatedDataLoss,
                mode: 'lines+markers',
                name: 'Repeated Web Data (Overfitting after 4 Epochs)',
                line: { color: '#b84a39', width: 3 },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>Validation Loss Degradation under High-Epoch Repetition (Epoch Caps)</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Training Epoch Count over Fixed Corpus', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Downstream Validation Loss', gridcolor: '#e5e5e5' },
            annotations: [
                {
                    x: 'Epoch 4',
                    y: 2.25,
                    text: '<b>Epoch Cap Threshold</b><br>Severe loss divergence begins',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 50,
                    ay: -35,
                    font: { size: 11, color: '#b84a39' }
                }
            ],
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture14Charts = function () {
        drawLshScurve('plotly-cs336-14-lsh-scurve');
        drawEpochCaps('plotly-cs336-14-epoch-caps');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture14Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-14-lsh-scurve')) {
        window.renderCs336Lecture14Charts();
    }
})();
