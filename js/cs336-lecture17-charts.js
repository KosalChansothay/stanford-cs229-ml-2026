/* ==========================================================================
   CS336 Lecture 17 interactive Plotly charts — Alignment — Multimodality.
   Chart 1: SigLIP Pairwise Binary Sigmoid Loss vs Softmax Contrastive Loss
   Chart 2: Vision Transformer (ViT) Patch Scaling & Token Count vs Resolution
   ========================================================================== */

(function () {
    'use strict';

    function drawSiglipLoss(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Compare SigLIP pairwise sigmoid loss to standard CLIP softmax cross-entropy
        // SigLIP loss for positive pair: -log sigma(t * u_i^T v_i - b)
        // For negative pair: -log sigma(-t * u_i^T v_j + b)
        var dotProducts = [];
        for (var d = -1.0; d <= 1.0; d += 0.05) dotProducts.push(d);

        var temp = 10.0;
        var bias = -10.0;

        var siglipPositiveLoss = dotProducts.map(function (d) {
            var logit = temp * d + bias;
            return -Math.log(1 / (1 + Math.exp(-logit)));
        });

        var siglipNegativeLoss = dotProducts.map(function (d) {
            var logit = -temp * d - bias;
            return -Math.log(1 / (1 + Math.exp(-logit)));
        });

        var data = [
            {
                x: dotProducts,
                y: siglipPositiveLoss,
                mode: 'lines',
                name: 'SigLIP Positive Pair Loss (Image-Caption Match)',
                line: { color: '#286b82', width: 3 }
            },
            {
                x: dotProducts,
                y: siglipNegativeLoss,
                mode: 'lines',
                name: 'SigLIP Negative Pair Loss (Unmatched Distractors)',
                line: { color: '#b84a39', width: 3 }
            }
        ];

        var layout = {
            title: { text: '<b>SigLIP Pairwise Binary Sigmoid Loss vs Dot Product Similarity (u^T v)</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Normalized Image-Text Dot Product (u^T v)', gridcolor: '#e5e5e5', range: [-1, 1] },
            yaxis: { title: 'Pairwise Loss Value', gridcolor: '#e5e5e5', range: [0, 15] },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawVlmTokens(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // AnyRes Dynamic Patch Slicing (14x14 ViT patches per 336x336 tile = 576 tokens/tile + 1 thumbnail tile)
        var gridConfigurations = ['1x1 (336×336)', '1x2 (336×672)', '2x2 (672×672)', '2x3 (672×1008)', '3x3 (1008×1008)', '4x4 (1344×1344)'];
        var tokenCounts = [576 * (1 + 1), 576 * (2 + 1), 576 * (4 + 1), 576 * (6 + 1), 576 * (9 + 1), 576 * (16 + 1)]; // including global thumbnail tile
        var relativeThroughput = [100, 72, 45, 32, 21, 11]; // % generation throughput relative to 1x1

        var data = [
            {
                x: gridConfigurations,
                y: tokenCounts,
                name: 'Visual Tokens Ingested by LLM',
                type: 'bar',
                yaxis: 'y1',
                marker: { color: '#286b82' }
            },
            {
                x: gridConfigurations,
                y: relativeThroughput,
                name: 'Relative Inference Throughput (%)',
                type: 'scatter',
                mode: 'lines+markers',
                yaxis: 'y2',
                line: { color: '#e69f00', width: 3 },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>AnyRes Vision Scaling: Ingested Visual Tokens vs Inference Throughput</b>', x: 0.5, font: { size: 15 } },
            xaxis: { gridcolor: '#e5e5e5' },
            yaxis: { title: 'Total Visual Tokens (576 per 336×336 Tile)', titlefont: { color: '#286b82' }, tickfont: { color: '#286b82' }, gridcolor: '#e5e5e5' },
            yaxis2: { title: 'Serving Throughput (%)', titlefont: { color: '#e69f00' }, tickfont: { color: '#e69f00' }, overlaying: 'y', side: 'right', range: [0, 110] },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 60, l: 65, r: 65 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture17Charts = function () {
        drawSiglipLoss('plotly-cs336-17-siglip-loss');
        drawVlmTokens('plotly-cs336-17-vlm-tokens');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture17Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-17-siglip-loss')) {
        window.renderCs336Lecture17Charts();
    }
})();
