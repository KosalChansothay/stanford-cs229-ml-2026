/* ==========================================================================
   Lecture 1 interactive Plotly charts.
   Loaded by courses/cs229/lecture-1.html AFTER the markdown is rendered,
   so the target <div> containers exist in the DOM by the time this runs.
   Requires Plotly.js (loaded from CDN in lecture-1.html).
   ========================================================================== */

(function () {
    'use strict';

    /* Shared PRNG so the synthetic housing data is stable across reloads. */
    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
            var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    /* ======================================================================
       Chart 1: House Price Prediction — Linear vs. Quadratic fit
       ====================================================================== */
    function drawHousePrice(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* 1. Synthetic housing data: price grows with size, plus noise. */
        var rand = mulberry32(7);
        var n = 40;
        var sizes = [], prices = [];
        for (var i = 0; i < n; i++) {
            var sqft = 500 + rand() * 2500;               // 500 – 3000 sq ft
            var price = 120 + 0.16 * sqft + 0.000022 * sqft * sqft   // mildly convex truth
                + (rand() - 0.5) * 160;                    // Gaussian-ish noise
            sizes.push(sqft);
            prices.push(price);
        }

        /* 2. Least-squares LINEAR fit: y = a + b*x (closed form). */
        var sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (var j = 0; j < n; j++) {
            sx += sizes[j]; sy += prices[j];
            sxx += sizes[j] * sizes[j]; sxy += sizes[j] * prices[j];
        }
        var b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        var a = (sy - b * sx) / n;

        /* 3. Least-squares QUADRATIC fit via normal equations (3x3). */
        var S = [0, 0, 0, 0, 0];        // sums of x^0..x^4
        var T = [0, 0, 0];              // sums of x^k * y
        for (var k = 0; k < n; k++) {
            var x = sizes[k], y2 = prices[k];
            S[0] += 1; S[1] += x; S[2] += x * x; S[3] += x * x * x; S[4] += x * x * x * x;
            T[0] += y2; T[1] += x * y2; T[2] += x * x * y2;
        }
        /* Solve [[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]] * c = T via Gaussian elimination. */
        function solve3(M, v) {
            var A = [
                [M[0][0], M[0][1], M[0][2], v[0]],
                [M[1][0], M[1][1], M[1][2], v[1]],
                [M[2][0], M[2][1], M[2][2], v[2]]
            ];
            for (var col = 0; col < 3; col++) {
                var piv = col;
                for (var r = col + 1; r < 3; r++) {
                    if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
                }
                var tmp = A[col]; A[col] = A[piv]; A[piv] = tmp;
                for (var r2 = col + 1; r2 < 3; r2++) {
                    var factor = A[r2][col] / A[col][col];
                    for (var c2 = col; c2 < 4; c2++) A[r2][c2] -= factor * A[col][c2];
                }
            }
            var sol = [0, 0, 0];
            for (var r3 = 2; r3 >= 0; r3--) {
                sol[r3] = (A[r3][3] - A[r3][0] * sol[0] - A[r3][1] * sol[1] - A[r3][2] * sol[2]) / A[r3][r3];
            }
            return sol;
        }
        var coef = solve3(
            [[S[0], S[1], S[2]], [S[1], S[2], S[3]], [S[2], S[3], S[4]]],
            [T[0], T[1], T[2]]
        );

        /* 4. Smooth fit curves. */
        var fitX = [], fitLin = [], fitQuad = [];
        for (var fx = 400; fx <= 3100; fx += 25) {
            fitX.push(fx);
            fitLin.push(a + b * fx);
            fitQuad.push(coef[0] + coef[1] * fx + coef[2] * fx * fx);
        }

        var data = [{
            x: sizes, y: prices, mode: 'markers', type: 'scatter',
            marker: { color: '#286b82', size: 9, line: { color: 'black', width: 1 } },
            name: 'Historical sales (training data)',
            hovertemplate: '%{x:,.0f} sq ft<br>$%{y:,.0f}k<extra></extra>'
        }, {
            x: fitX, y: fitLin, mode: 'lines', type: 'scatter',
            line: { color: '#d16a1a', width: 3, dash: 'dash' },
            name: 'Linear fit: h(x) = θ₀ + θ₁x'
        }, {
            x: fitX, y: fitQuad, mode: 'lines', type: 'scatter',
            line: { color: '#7a1f5c', width: 3 },
            name: 'Quadratic fit: h(x) = θ₀ + θ₁x + θ₂x²'
        }];

        Plotly.newPlot(el, data, {
            title: { text: '<b>House Price Prediction (Regression)</b><br><sup>Supervised learning: continuous output y = price</sup>', x: 0.5 },
            xaxis: { title: 'Living area — square feet (feature x)', gridcolor: 'rgba(15, 23, 42, 0.08)', range: [300, 3200] },
            yaxis: { title: 'Price — $ thousands (target y)', gridcolor: 'rgba(15, 23, 42, 0.08)' },
            height: 540,
            hovermode: 'closest',
            responsive: true,
            autosize: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.85)', bordercolor: 'rgba(15, 23, 42, 0.12)', borderwidth: 1 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { l: 65, r: 35, t: 85, b: 65 }
        }, { displayModeBar: false, responsive: true });
    }

    /* ======================================================================
       Chart 2: Townhouse vs. Single-Family House — Decision Boundary
       ====================================================================== */
    function drawClassification(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        var isDark = document.documentElement.getAttribute('data-theme') === 'dark';

        /* 1. Synthetic 2D data: two distinct classes separated by a linear boundary.
              Andrew Ng CS229 Lecture 1: Predicting housing type (Townhouse y=0 vs Single-Family y=1)
              Features:
                x1 = lot size (in thousands of sq ft)
                x2 = living area (in sq ft)
              Linear decision boundary: theta^T x = 0 <=> x2 = 2800 - 220 * x1
        */
        var rand = mulberry32(101);
        var townhouses = [];
        var houses = [];

        // 26 Townhouses: smaller lot & compact area (theta^T x < 0)
        while (townhouses.length < 26) {
            var lot = 1.8 + rand() * 4.4; // 1.8k - 6.2k
            var sqft = 650 + rand() * 1350; // 650 - 2000
            var boundaryY = 2800 - 220 * lot;
            if (sqft < boundaryY - 70) {
                townhouses.push([+lot.toFixed(2), Math.round(sqft)]);
            }
        }

        // 26 Single-family houses: larger lot & larger area (theta^T x > 0)
        while (houses.length < 26) {
            var lot = 3.6 + rand() * 6.5; // 3.6k - 10.1k
            var sqft = 1350 + rand() * 1750; // 1350 - 3100
            var boundaryY = 2800 - 220 * lot;
            if (sqft > boundaryY + 70 && sqft <= 3150 && lot <= 10.2) {
                houses.push([+lot.toFixed(2), Math.round(sqft)]);
            }
        }

        /* 2. Linear Decision Boundary Line across the full visible domain [1.5, 10.5] */
        var lineLot = [1.5, 10.5];
        var lineSqft = lineLot.map(function (lot) {
            return 2800 - 220 * lot; // from 2470 down to 490
        });

        var badgeBg = isDark ? 'rgba(15, 23, 42, 0.88)' : 'rgba(255, 255, 255, 0.92)';
        var badgeBorder = isDark ? 'rgba(255, 255, 255, 0.16)' : 'rgba(15, 23, 42, 0.12)';
        var boundaryColor = isDark ? '#F1F5F9' : '#0F172A';

        var data = [
            // Class 1: Single-family houses (Upper right)
            {
                x: houses.map(function (p) { return p[0]; }),
                y: houses.map(function (p) { return p[1]; }),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: '#2563EB',
                    size: 11,
                    symbol: 'triangle-up',
                    line: { color: '#1D4ED8', width: 1.5 }
                },
                name: 'Single-family house (y = 1)',
                hovertemplate: '<b>Single-Family House</b><br>• Lot size: %{x:.2f}k sq ft<br>• Living area: %{y:,.0f} sq ft<br>• Label: y = 1 (θᵀx > 0)<extra></extra>'
            },
            // Class 0: Townhouses (Lower left)
            {
                x: townhouses.map(function (p) { return p[0]; }),
                y: townhouses.map(function (p) { return p[1]; }),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: '#E11D48',
                    size: 10,
                    symbol: 'circle',
                    line: { color: '#BE123C', width: 1.5 }
                },
                name: 'Townhouse (y = 0)',
                hovertemplate: '<b>Townhouse</b><br>• Lot size: %{x:.2f}k sq ft<br>• Living area: %{y:,.0f} sq ft<br>• Label: y = 0 (θᵀx < 0)<extra></extra>'
            },
            // Linear decision boundary
            {
                x: lineLot,
                y: lineSqft,
                mode: 'lines',
                type: 'scatter',
                line: {
                    color: boundaryColor,
                    width: 3,
                    dash: 'dash'
                },
                name: 'Decision boundary (θᵀx = 0)',
                hoverinfo: 'skip'
            }
        ];

        var layout = {
            title: {
                text: '<b>Townhouse vs. Single-Family House (Classification)</b><br><sup>Supervised learning: linear decision boundary θᵀx = 0 separates the 2D feature space</sup>',
                x: 0.5,
                font: { size: 16 }
            },
            xaxis: {
                title: 'Lot size — thousands of sq ft (x₁)',
                gridcolor: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)',
                range: [1.5, 10.5],
                zeroline: false
            },
            yaxis: {
                title: 'Living area — sq ft (x₂)',
                gridcolor: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)',
                range: [400, 3200],
                zeroline: false
            },
            height: 540,
            hovermode: 'closest',
            responsive: true,
            autosize: true,
            legend: {
                x: 0.02,
                y: 0.98,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: badgeBg,
                bordercolor: badgeBorder,
                borderwidth: 1,
                font: { size: 12 }
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { l: 65, r: 35, t: 85, b: 65 },
            shapes: [
                // Half-space 1: Single-family region (upper-right)
                {
                    type: 'path',
                    path: 'M 1.5 2470 L 10.5 490 L 10.5 3200 L 1.5 3200 Z',
                    fillcolor: isDark ? 'rgba(37, 99, 235, 0.14)' : 'rgba(37, 99, 235, 0.07)',
                    line: { width: 0 },
                    layer: 'below'
                },
                // Half-space 2: Townhouse region (lower-left)
                {
                    type: 'path',
                    path: 'M 1.5 400 L 10.5 400 L 10.5 490 L 1.5 2470 Z',
                    fillcolor: isDark ? 'rgba(225, 29, 72, 0.13)' : 'rgba(225, 29, 72, 0.06)',
                    line: { width: 0 },
                    layer: 'below'
                }
            ],
            annotations: [
                // Zone label: Single-Family
                {
                    x: 8.6,
                    y: 2850,
                    xref: 'x',
                    yref: 'y',
                    text: '<b style="color:#2563EB;">Predicted: Single-Family (ŷ = 1)</b><br><span style="font-size:11px;opacity:0.85;">Half-space θᵀx > 0</span>',
                    showarrow: false,
                    align: 'center',
                    bgcolor: badgeBg,
                    bordercolor: 'rgba(37, 99, 235, 0.35)',
                    borderwidth: 1,
                    borderpad: 6
                },
                // Zone label: Townhouse
                {
                    x: 3.2,
                    y: 720,
                    xref: 'x',
                    yref: 'y',
                    text: '<b style="color:#E11D48;">Predicted: Townhouse (ŷ = 0)</b><br><span style="font-size:11px;opacity:0.85;">Half-space θᵀx < 0</span>',
                    showarrow: false,
                    align: 'center',
                    bgcolor: badgeBg,
                    bordercolor: 'rgba(225, 29, 72, 0.35)',
                    borderwidth: 1,
                    borderpad: 6
                },
                // Decision boundary pointer
                {
                    x: 5.8,
                    y: 1524,
                    xref: 'x',
                    yref: 'y',
                    text: '<b>Decision Boundary: θᵀx = 0</b>',
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 1.5,
                    arrowcolor: boundaryColor,
                    ax: 55,
                    ay: -40,
                    bgcolor: badgeBg,
                    bordercolor: badgeBorder,
                    borderwidth: 1,
                    borderpad: 5,
                    font: { size: 11, color: isDark ? '#F8FAFC' : '#0F172A' }
                }
            ]
        };

        Plotly.newPlot(el, data, layout, { displayModeBar: false, responsive: true });
    }

    /* ======================================================================
       Public entry point — called once the markdown (and chart containers)
       has been injected into the DOM by js/script.js.
       ====================================================================== */
    window.renderLecture1Charts = function () {
        drawHousePrice('plotly-house-price');
        drawClassification('plotly-house-classification');
    };

    /* Charts live inside the fetched markdown, so wait for the render event.
       The readyState guard covers the no-JS-markdown / cached case. */
    document.addEventListener('markdown:rendered', window.renderLecture1Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-house-price')) {
        window.renderLecture1Charts();
    }

    /* Re-render charts when theme changes to update dark/light specific annotations and shapes */
    document.addEventListener('theme:changed', function () {
        if (document.getElementById('plotly-house-classification')) {
            drawClassification('plotly-house-classification');
        }
    });
})();
