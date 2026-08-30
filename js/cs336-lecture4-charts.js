/* ==========================================================================
   CS336 Lecture 4 interactive Plotly charts — Attention Alternatives & MoE.
   Chart 1: Mixture of Experts (MoE) Top-2 Routing & Expert Load Distribution
   Chart 2: Memory Scaling: Standard Attention O(T^2) vs Linear SSM O(T)
   ========================================================================== */

(function () {
    'use strict';

    function drawMoeRouting(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // 8 Experts. Simulate token assignment counts with vs without auxiliary load-balancing loss
        var experts = ['Expert 0', 'Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5', 'Expert 6', 'Expert 7'];
        // Without aux loss: routing collapses to popular experts (Expert 1, 4 starved, 0, 2 overloaded)
        var collapsedCounts = [420, 18, 510, 35, 12, 380, 45, 80];
        // With aux loss (loss_aux = alpha * sum(f_i * P_i)): balanced across experts
        var balancedCounts = [190, 185, 198, 182, 195, 190, 188, 192];

        var data = [
            {
                x: experts,
                y: collapsedCounts,
                name: 'Without Aux Loss (Load Collapse)',
                type: 'bar',
                marker: { color: '#b84a39' }
            },
            {
                x: experts,
                y: balancedCounts,
                name: 'With Aux Load-Balancing Loss (Balanced)',
                type: 'bar',
                marker: { color: '#286b82' }
            }
        ];

        var layout = {
            title: { text: '<b>MoE Expert Token Allocation: Collapse vs Balanced Routing</b>', x: 0.5, font: { size: 15 } },
            barmode: 'group',
            xaxis: { title: 'Mixture-of-Experts (8 Experts)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'Tokens Routed per Batch (Total = 1500)', gridcolor: '#e5e5e5' },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawLinearAttentionMemory(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var seqLengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536];
        var xLabels = ['1k', '2k', '4k', '8k', '16k', '32k', '64k'];

        // Attention matrix memory: N_layers * N_heads * T^2 * 2 bytes (FP16)
        // For 32 layers, 32 heads: 1024 * T^2 * 2 bytes
        var standardAttnMb = seqLengths.map(function (T) {
            return (32 * 32 * T * T * 2) / (1024 * 1024 * 1024); // in GB
        });

        // Linear Attention / SSM recurrent state: constant per token O(d^2) state
        // Recurrent state size = 32 layers * (d_state * d_model) = constant across T during inference
        var ssmMb = seqLengths.map(function (T) {
            return 0.15; // Constant ~150 MB state
        });

        var data = [
            {
                x: xLabels,
                y: standardAttnMb,
                mode: 'lines+markers',
                name: 'Standard Softmax Attention O(T²)',
                line: { color: '#b84a39', width: 3 },
                marker: { size: 7 }
            },
            {
                x: xLabels,
                y: ssmMb,
                mode: 'lines+markers',
                name: 'SSM / Linear Attention Recurrent State O(1)',
                line: { color: '#286b82', width: 3 },
                marker: { size: 7 }
            }
        ];

        var layout = {
            title: { text: '<b>Inference Memory Footprint: Standard Attention vs SSM / Linear Recurrence</b>', x: 0.5, font: { size: 15 } },
            xaxis: { title: 'Sequence Length (Tokens)', gridcolor: '#e5e5e5' },
            yaxis: { title: 'State Memory Required (GB)', gridcolor: '#e5e5e5', type: 'log' },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 55, l: 65, r: 35 },
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture4Charts = function () {
        drawMoeRouting('plotly-cs336-4-moe-routing');
        drawLinearAttentionMemory('plotly-cs336-4-linear-attention');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture4Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-4-moe-routing')) {
        window.renderCs336Lecture4Charts();
    }
})();
