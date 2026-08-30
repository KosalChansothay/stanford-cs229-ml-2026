/* ==========================================================================
   CS336 Lecture 13 interactive Plotly charts — Data (Sources & Datasets).
   Chart 1: Pre-training Data Pipeline Volume Waterfall (Common Crawl Filtering)
   Chart 2: Pre-training Domain Mix Distribution & Token Proportions
   ========================================================================== */

(function () {
    'use strict';

    function drawDataWaterfall(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Common Crawl data processing funnel (in Petabytes / Terabytes)
        var stages = ['Raw Common Crawl (Warc)', 'HTML Text Extracted', 'Language Filtered (FastText)', 'Quality Filtered (Classifier)', 'Exact / MinHash Deduplicated', 'Curated Pre-training Corpus'];
        var volumes = [4500, 380, 240, 110, 48, 30]; // Terabytes

        var data = [{
            x: stages,
            y: volumes,
            type: 'bar',
            marker: {
                color: ['#1f3a52', '#286b82', '#519db8', '#e69f00', '#b84a39', '#3d8b57']
            },
            text: volumes.map(function (v) { return v >= 1000 ? (v / 1000).toFixed(1) + ' PB' : v + ' TB'; }),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Volume: %{text}<extra></extra>'
        }];

        var layout = {
            title: { text: '<b>Pre-training Data Filtering Funnel: Volume Retention Waterfall</b>', x: 0.5, font: { size: 15 } },
            xaxis: { gridcolor: '#e5e5e5' },
            yaxis: { title: 'Corpus Volume (Terabytes, Log Scale)', gridcolor: '#e5e5e5', type: 'log' },
            height: 450,
            responsive: true,
            plot_bgcolor: 'white',
            margin: { t: 70, b: 70, l: 65, r: 35 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    function drawDomainMix(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // Frontier pre-training token mixture proportions (e.g. LLaMA-3 / FineWeb style 15T tokens)
        var domains = ['General Web (FineWeb/C4)', 'Code (GitHub/StackOverflow)', 'STEM & Math (arXiv/Papers)', 'Books & Long-Form', 'Encyclopedic (Wikipedia)', 'Synthetic Reasoning & Multi-turn'];
        var tokenPercentages = [50, 20, 12, 6, 4, 8];
        var colors = ['#286b82', '#3d8b57', '#e69f00', '#7a5195', '#519db8', '#b84a39'];

        var data = [{
            labels: domains,
            values: tokenPercentages,
            type: 'pie',
            hole: 0.45,
            marker: { colors: colors },
            textinfo: 'label+percent',
            textposition: 'outside',
            hovertemplate: '<b>%{label}</b><br>Token Proportion: %{value}%<extra></extra>'
        }];

        var layout = {
            title: { text: '<b>Representative Pre-training Domain Token Mixture (15T Tokens)</b>', x: 0.5, font: { size: 15 } },
            height: 460,
            responsive: true,
            showlegend: false,
            margin: { t: 70, b: 40, l: 40, r: 40 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture13Charts = function () {
        drawDataWaterfall('plotly-cs336-13-data-waterfall');
        drawDomainMix('plotly-cs336-13-domain-mix');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture13Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-13-data-waterfall')) {
        window.renderCs336Lecture13Charts();
    }
})();
