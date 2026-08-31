/* ==========================================================================
   CS336 Lecture 11 interactive Plotly charts — Scaling Laws & Advanced Optimizers.
   Chart 1: Maximal Update Parameterization (muP) Zero-Shot LR Transfer vs SP
   Chart 2: Warmup-Stable-Decay (WSD) vs Cosine Annealing Learning Rate Schedule
   ========================================================================== */

(function () {
    'use strict';

    function drawMupTransfer(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var widths = [256, 512, 1024, 2048, 4096, 8192];
        var xLabels = widths.map(function (w) { return 'd=' + w; });

        // Under Standard Parameterization (SP), optimal LR shifts downward as 1/width
        var spOptimalLr = [0.004, 0.0028, 0.0019, 0.0012, 0.0008, 0.0005];
        // Under muP (Maximal Update Parameterization), optimal LR is invariant across width
        var mupOptimalLr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002];

        var data = [
            {
                x: xLabels,
                y: mupOptimalLr,
                mode: 'lines+markers',
                name: 'μP (Maximal Update Parameterization: Zero-Shot Transfer)',
                line: { color: '#286b82', width: 3.5 },
                marker: { size: 8 }
            },
            {
                x: xLabels,
                y: spOptimalLr,
                mode: 'lines+markers',
                name: 'Standard Parameterization SP (LR Shifts with Width)',
                line: { color: '#b84a39', width: 2.5, dash: 'dash' },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>Hyperparameter Transfer: Optimal Learning Rate vs Model Width</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Hidden Dimension (Model Width d_model)', gridcolor: '#e5e5e5' },
            yaxis: {
                title: 'Optimal Learning Rate (η*)',
                gridcolor: '#e5e5e5',
                type: 'log',
                tickmode: 'array',
                tickvals: [0.0005, 0.001, 0.002, 0.004, 0.008],
                ticktext: ['5e-4', '1e-3', '2e-3 (μP)', '4e-3', '8e-3']
            },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawWsdSchedule(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var steps = [];
        for (var s = 0; s <= 100; s += 1) steps.push(s);

        // Cosine decay schedule over 100 steps
        var cosineLr = steps.map(function (s) {
            if (s <= 5) return (s / 5) * 1.0;
            return 0.5 * (1 + Math.cos(Math.PI * (s - 5) / 95));
        });

        // WSD (Warmup-Stable-Decay): Warmup 5%, Flat Stable 80%, Anneal 15%
        var wsdLr = steps.map(function (s) {
            if (s <= 5) return (s / 5) * 1.0;
            if (s <= 85) return 1.0;
            return 1.0 - (s - 85) / 15 * 0.9;
        });

        var data = [
            {
                x: steps,
                y: wsdLr,
                mode: 'lines',
                name: 'WSD Schedule (Constant High Stable Phase + 15% Decay)',
                line: { color: '#286b82', width: 3 }
            },
            {
                x: steps,
                y: cosineLr,
                mode: 'lines',
                name: 'Cosine Schedule (Fixed Horizon Decay)',
                line: { color: '#b84a39', width: 2.5, dash: 'dash' }
            }
        ];

        var layout = {
            title: { text: '<b>Learning Rate Schedule: Warmup-Stable-Decay (WSD) vs Cosine Annealing</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Training Progress (% of Total Steps)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Relative Learning Rate Factor', gridcolor: '#e5e5e5', range: [0, 1.1] },
            annotations: [
                {
                    x: 85,
                    y: 1.0,
                    text: '<b>Stable Phase</b><br>(Branch checkpoints at any step)',
                    showarrow: true,
                    arrowhead: 2,
                    ax: -60,
                    ay: -30,
                    font: { size: 10, color: '#286b82' }
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

    window.renderCs336Lecture11Charts = function () {
        drawMupTransfer('plotly-cs336-11-mup-transfer');
        drawWsdSchedule('plotly-cs336-11-wsd-schedule');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture11Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-11-mup-transfer')) {
        window.renderCs336Lecture11Charts();
    }
})();
