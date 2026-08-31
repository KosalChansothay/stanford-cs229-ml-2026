/* ==========================================================================
   CS336 Lecture 15 interactive Plotly charts — Post-Training (SFT & DPO).
   Chart 1: Direct Preference Optimization (DPO) Implicit Reward Loss Gradient
   Chart 2: SFT Mode Extraction: Base Pre-training vs Post-Trained Response Distribution
   ========================================================================== */

(function () {
    'use strict';

    function drawDpoLoss(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // DPO loss: L_DPO = -log sigma(beta * log(pi(y_w|x)/pi_ref(y_w|x)) - beta * log(pi(y_l|x)/pi_ref(y_l|x)))
        // Let delta_r = r_theta(x, y_w) - r_theta(x, y_l) (Implicit reward margin)
        var margins = [];
        for (var m = -4; m <= 4; m += 0.1) margins.push(m);

        var beta = 0.1;
        var dpoLosses = margins.map(function (m) {
            return -Math.log(1 / (1 + Math.exp(-m)));
        });

        var gradientWeights = margins.map(function (m) {
            var sigma = 1 / (1 + Math.exp(m)); // sigma(-margin)
            return sigma;
        });

        var data = [
            {
                x: margins,
                y: dpoLosses,
                name: 'DPO Loss -log σ(Δr)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#b84a39', width: 3 }
            },
            {
                x: margins,
                y: gradientWeights,
                name: 'Gradient Update Scale σ(-Δr)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#286b82', width: 3, dash: 'dash' }
            }
        ];

        var layout = {
            title: { text: '<b>Direct Preference Optimization (DPO): Loss & Gradient Weighting</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Implicit Reward Margin: r_θ(y_w) - r_θ(y_l)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Loss / Gradient Magnitude', gridcolor: '#e5e5e5' },
            annotations: [
                {
                    x: -2.5,
                    y: 2.6,
                    text: '<b>High Penalty Zone</b><br>Model prefers rejected answer y_l',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 50,
                    ay: -25,
                    font: { size: 10, color: '#b84a39' }
                },
                {
                    x: 2.5,
                    y: 0.1,
                    text: '<b>Vanishing Gradient</b><br>Model correctly prefers chosen y_w',
                    showarrow: true,
                    arrowhead: 2,
                    ax: -50,
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

    function drawSftMode(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Categorical simulation of response formats
        var formats = ['Direct Answer', 'Chain-of-Thought Explanation', 'Raw Web Scraping Noise', 'Unformatted Dialogue', 'Code Block + Test'];
        var baseProb = [0.15, 0.10, 0.45, 0.20, 0.10];
        var sftProb = [0.40, 0.35, 0.00, 0.05, 0.20];

        var data = [
            {
                x: formats,
                y: baseProb.map(function (p) { return p * 100; }),
                name: 'Base Pre-trained Model (Diffuse Distribution)',
                type: 'bar',
                marker: { color: '#999' }
            },
            {
                x: formats,
                y: sftProb.map(function (p) { return p * 100; }),
                name: 'Post-Trained / SFT Model (Mode Extraction)',
                type: 'bar',
                marker: { color: '#286b82' }
            }
        ];

        var layout = {
            title: { text: '<b>SFT as Mode Extraction: Probability Distribution Shift on User Prompts</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            barmode: 'group',
            xaxis: { gridcolor: '#e5e5e5' },
            yaxis: { title: 'Probability of Output Style (%)', gridcolor: '#e5e5e5' },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture15Charts = function () {
        drawDpoLoss('plotly-cs336-15-dpo-loss');
        drawSftMode('plotly-cs336-15-sft-mode');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture15Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-15-dpo-loss')) {
        window.renderCs336Lecture15Charts();
    }
})();
