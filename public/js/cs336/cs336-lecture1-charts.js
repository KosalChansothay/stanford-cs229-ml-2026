/* ==========================================================================
   CS336 Lecture 1 interactive Plotly charts — BPE Tokenization.
   Chart 1: BPE compression trace — token count shrinks as merges are
            applied to a toy corpus (real BPE training, not decorative).
   Chart 2: Attention cost — quadratic O(L^2) savings from compression.
   Loaded by courses/cs336/lecture-1.html AFTER the markdown is rendered.
   Requires Plotly.js (loaded from CDN in lecture-1.html).
   ========================================================================== */

(function () {
    'use strict';

    /* Run real BPE training on a corpus; return {steps, counts} where
       counts[t] is the token-sequence length after m merges. */
    function runBpe(text, numMerges) {
        var ids = [];
        for (var i = 0; i < text.length; i++) ids.push(text.charCodeAt(i));

        var steps = [0], tokenCounts = [ids.length];
        for (var m = 0; m < numMerges; m++) {
            var counts = {};
            for (var i = 0; i < ids.length - 1; i++) {
                var key = ids[i] + ',' + ids[i + 1];
                counts[key] = (counts[key] || 0) + 1;
            }
            var best = null, bestC = 1; /* require count >= 2 */
            for (var k in counts) {
                if (counts[k] > bestC) { bestC = counts[k]; best = k; }
            }
            if (best === null) break;

            var parts = best.split(',').map(Number);
            var out = [], j = 0;
            while (j < ids.length) {
                if (j < ids.length - 1 && ids[j] === parts[0] && ids[j + 1] === parts[1]) {
                    out.push(256 + m); j += 2;
                } else {
                    out.push(ids[j]); j += 1;
                }
            }
            ids = out;
            steps.push(m + 1);
            tokenCounts.push(ids.length);
        }
        return { steps: steps, counts: tokenCounts };
    }

    /* ======================================================================
       Chart 1: BPE compression trace on a toy corpus.
       ====================================================================== */
    function drawBpeMerges(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var corpus = 'the cat in the hat the cat sat on the mat the cat and the hat';
        var result = runBpe(corpus, 60);

        var vocabSizes = result.steps.map(function (m) { return 256 + m; });

        var data = [{
            x: result.steps,
            y: result.counts,
            mode: 'lines+markers',
            name: 'Sequence length (tokens)',
            line: { color: '#286b82', width: 3 },
            marker: { size: 6 },
            hovertemplate: 'merges=%{x} (vocab %{customdata})<br>%{y} tokens<extra></extra>',
            customdata: result.steps.map(function (m) { return 256 + m; })
        }];

        var layout = {
            title: { text: '<b>BPE Training: Token Count Shrinks with Each Merge</b>', x: 0.5, y: 0.98, font: { size: 16 } },
            xaxis: { title: 'Merge operations M', gridcolor: 'lightgray' },
            yaxis: { title: 'Sequence length (tokens)', gridcolor: 'lightgray' },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: Quadratic attention savings from compression.
       ====================================================================== */
    function drawAttentionSavings(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var rawBytes = 1000;
        var ratios = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
        var labels = ratios.map(function (r) { return r.toFixed(1) + 'x'; });
        var costs = ratios.map(function (r) {
            var L = rawBytes / r;
            return L * L;
        });

        var data = [{
            x: labels,
            y: costs,
            type: 'bar',
            name: 'Attention compute O(L²)',
            marker: { color: '#286b82' },
            hovertemplate: 'compression %{x}: %{y:,} units<extra></extra>'
        }];

        var layout = {
            title: { text: '<b>Why Tokenize? Quadratic Attention Savings</b>', x: 0.5, y: 0.98, font: { size: 16 } },
            xaxis: { title: 'Compression ratio (bytes per token)', gridcolor: 'lightgray' },
            yaxis: { title: 'Relative attention compute', gridcolor: 'lightgray' },
            height: 450,
            responsive: true,
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90, b: 55, l: 70, r: 50 }
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false });
    }

    window.renderCs336Lecture1Charts = function () {
        drawBpeMerges('plotly-cs336-1-bpe-merges');
        drawAttentionSavings('plotly-cs336-1-attention-savings');
    };

    document.addEventListener('markdown:rendered', window.renderCs336Lecture1Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs336-1-bpe-merges')) {
        window.renderCs336Lecture1Charts();
    }
})();
