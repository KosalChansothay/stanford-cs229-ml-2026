/* ==========================================================================
   CS336 Lecture 12 interactive Plotly charts — Evaluation (Methodology & Contamination).
   Chart 1: Chatbot Arena Bradley-Terry Win Probability vs Elo Rating Difference
   Chart 2: LLM-as-a-Judge Length Bias vs True Win Rate
   ========================================================================== */

(function () {
    'use strict';

    function drawBradleyTerry(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Bradley-Terry logistic model: P(A > B) = 1 / (1 + 10^((R_B - R_A) / 400))
        var eloDiffs = [];
        for (var d = -400; d <= 400; d += 10) eloDiffs.push(d);

        var winProbs = eloDiffs.map(function (diff) {
            return 1 / (1 + Math.pow(10, -diff / 400));
        });

        var data = [{
            x: eloDiffs,
            y: winProbs,
            mode: 'lines',
            name: 'P(Model A beats Model B)',
            line: { color: '#286b82', width: 3.5 }
        }];

        var layout = {
            title: { text: '<b>Chatbot Arena: Bradley-Terry Expected Win Rate vs Elo Difference (R_A - R_B)</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Elo Rating Advantage (R_A - R_B)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Probability of Model A Winning', gridcolor: '#e5e5e5', range: [0, 1] },
            annotations: [
                {
                    x: 100,
                    y: 1 / (1 + Math.pow(10, -100 / 400)),
                    text: '<b>+100 Elo Diff</b><br>Win Rate = 64.0%',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 50,
                    ay: 30,
                    font: { size: 11, color: '#286b82' }
                },
                {
                    x: 200,
                    y: 1 / (1 + Math.pow(10, -200 / 400)),
                    text: '<b>+200 Elo Diff</b><br>Win Rate = 75.9%',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 50,
                    ay: 30,
                    font: { size: 11, color: '#286b82' }
                }
            ],
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawJudgeLengthBias(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var lengthRatios = ['0.5x (Concise)', '0.8x (Compact)', '1.0x (Equal)', '1.5x (Verbose)', '2.0x (Bloated)'];
        // Uncalibrated LLM Judge win rates on identical content with padded formatting
        var uncalibratedWins = [28, 41, 50, 68, 82];
        // Length-controlled debiased win rates
        var calibratedWins = [50, 50, 50, 50, 50];

        var data = [
            {
                x: lengthRatios,
                y: uncalibratedWins,
                name: 'Uncalibrated LLM Judge (Spurious Verbosity Bias)',
                type: 'bar',
                marker: { color: '#b84a39' }
            },
            {
                x: lengthRatios,
                y: calibratedWins,
                name: 'Length-Controlled Debiased Judge (True Quality)',
                type: 'bar',
                marker: { color: '#286b82' }
            }
        ];

        var layout = {
            title: { text: '<b>LLM-as-a-Judge Length Bias: Win Rate Distortion on Equivalent Answers</b>', x: 0.5, font: { size: 15 } },
            barmode: 'group',
            xaxis: { title: 'Response Length Relative to Baseline', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Evaluated Win Rate (%)', gridcolor: '#e5e5e5', range: [0, 100] },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture12Charts = function () {
        drawBradleyTerry('plotly-cs336-12-bradley-terry');
        drawJudgeLengthBias('plotly-cs336-12-judge-length-bias');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture12Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-12-bradley-terry')) {
        window.renderCs336Lecture12Charts();
    }
})();
