/* ==========================================================================
   CS336 Lecture 16 interactive Plotly charts — Post-Training — RLVR.
   Chart 1: Group Relative Policy Optimization (GRPO) Normalized Advantage Distribution
   Chart 2: Test-Time Compute Scaling: Pass@k and Majority Voting vs Rollouts N
   ========================================================================== */

(function () {
    'use strict';

    function drawGrpoAdvantage(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Group of G=8 completions for a math prompt with binary reward {0, 1}
        var completions = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6', 'Sample 7', 'Sample 8'];
        var rawRewards = [1, 0, 1, 0, 1, 0, 0, 1]; // 4 correct, 4 incorrect (mean = 0.5, std = 0.5345)
        var mean = 0.5;
        var std = 0.5345;

        var normalizedAdvantages = rawRewards.map(function (r) {
            return (r - mean) / (std + 1e-4);
        });

        var colors = normalizedAdvantages.map(function (a) {
            return a > 0 ? '#3d8b57' : '#b84a39';
        });

        var data = [
            {
                x: completions,
                y: normalizedAdvantages,
                type: 'bar',
                marker: { color: colors },
                text: normalizedAdvantages.map(function (a) { return (a > 0 ? '+' : '') + a.toFixed(2); }),
                textposition: 'auto',
                hovertemplate: '<b>%{x}</b><br>GRPO Advantage: %{y:.2f}<extra></extra>'
            }
        ];

        var layout = {
            title: { text: '<b>GRPO Advantage Normalization (A_i = (r_i - μ) / σ) across G=8 Rollouts</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Sampled Completion Rollouts (Group Size G = 8)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Normalized Advantage (A_i)', gridcolor: '#e5e5e5', range: [-1.5, 1.5] },
            annotations: [
                {
                    x: 'Sample 1',
                    y: 1.0,
                    text: '<b>Positive Reinforcement</b><br>(Correct Proof)',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 40,
                    ay: -25,
                    font: { size: 10, color: '#3d8b57' }
                },
                {
                    x: 'Sample 2',
                    y: -1.0,
                    text: '<b>Policy Penalization</b><br>(Incorrect Calculation)',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 40,
                    ay: 25,
                    font: { size: 10, color: '#b84a39' }
                }
            ],
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawTestTimeCompute(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var sampleCounts = [1, 2, 4, 8, 16, 32, 64];
        var xLabels = sampleCounts.map(function (n) { return 'N=' + n; });

        // Pass@N accuracy: 1 - (1 - p)^N where single-pass p = 0.25 (MATH benchmark)
        var passAtN = sampleCounts.map(function (N) {
            return (1 - Math.pow(1 - 0.25, N)) * 100;
        });

        // Majority voting accuracy across N sampled rollouts
        var majVote = [25.0, 31.5, 42.0, 53.2, 62.8, 69.5, 74.0];

        var data = [
            {
                x: xLabels,
                y: passAtN,
                mode: 'lines+markers',
                name: 'Pass@N with Verifier Oracle',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            },
            {
                x: xLabels,
                y: majVote,
                mode: 'lines+markers',
                name: 'Majority Voting (Self-Consistency Consensus)',
                line: { color: '#3d8b57', width: 3 },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>Test-Time Compute Scaling: Math Benchmark Accuracy vs Sampled Rollouts N</b>', x: 0.5, y: 0.98, font: { size: 15 } },
            xaxis: { title: 'Test-Time Search Budget (Number of Candidate Rollouts N)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Benchmark Accuracy (%)', gridcolor: '#e5e5e5', range: [20, 100] },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 },
            legend: { orientation: 'h', y: 1.16, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture16Charts = function () {
        drawGrpoAdvantage('plotly-cs336-16-grpo-advantage');
        drawTestTimeCompute('plotly-cs336-16-test-time-compute');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture16Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-16-grpo-advantage')) {
        window.renderCs336Lecture16Charts();
    }
})();
