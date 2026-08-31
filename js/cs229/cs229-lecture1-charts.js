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
            xaxis: { title: 'Living area — square feet (feature x)', gridcolor: 'lightgray', range: [300, 3200] },
            yaxis: { title: 'Price — $ thousands (target y)', gridcolor: 'lightgray' },
            width: 780,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'lightgray', borderwidth: 1 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90 }
        }, { displayModeBar: false });
    }

    /* ======================================================================
       Chart 2: Townhouse vs. Single-Family House — Decision Boundary
       ====================================================================== */
    function drawClassification(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        /* 1. Synthetic 2D data: two classes separated by a linear boundary.
              Boundary: 0.9*lot + 1.1*sqft - 2600 = 0  (theta^T x + theta0). */
        var rand = mulberry32(21);
        var n = 45;
        var theta0 = -2600, theta1 = 0.9, theta2 = 1.1;
        var houses = [], townhouses = [];
        for (var i = 0; i < n; i++) {
            var lot = 2 + rand() * 8;          // lot size, 1000s of sq ft
            var sqft = 600 + rand() * 2400;    // living area, sq ft
            var score = theta0 + theta1 * lot * 1000 + theta2 * sqft;
            var pt = [lot, sqft];
            if (score >= 0) houses.push(pt);
            else townhouses.push(pt);
        }

        /* 2. Decision boundary line across the plot range. */
        var lineLot = [2, 10];
        var lineSqft = lineLot.map(function (lot) {
            return -(theta0 + theta1 * lot * 1000) / theta2;
        });

        var data = [{
            x: houses.map(function (p) { return p[0]; }),
            y: houses.map(function (p) { return p[1]; }),
            mode: 'markers', type: 'scatter',
            marker: { color: 'crimson', size: 11, symbol: 'triangle-up', line: { color: 'black', width: 1 } },
            name: 'Single-family house (y = 1)',
            hovertemplate: 'Lot: %{x:.1f}k sq ft<br>Area: %{y:,.0f} sq ft<extra></extra>'
        }, {
            x: townhouses.map(function (p) { return p[0]; }),
            y: townhouses.map(function (p) { return p[1]; }),
            mode: 'markers', type: 'scatter',
            marker: { color: 'dodgerblue', size: 11, symbol: 'circle', line: { color: 'black', width: 1 } },
            name: 'Townhouse (y = 0)',
            hovertemplate: 'Lot: %{x:.1f}k sq ft<br>Area: %{y:,.0f} sq ft<extra></extra>'
        }, {
            x: lineLot, y: lineSqft, mode: 'lines', type: 'scatter',
            line: { color: 'black', width: 3, dash: 'dash' },
            name: 'Decision boundary (θᵀx = 0)',
            hoverinfo: 'skip'
        }];

        Plotly.newPlot(el, data, {
            title: { text: '<b>Townhouse vs. Single-Family House (Classification)</b><br><sup>Supervised learning: discrete label y, linear decision boundary</sup>', x: 0.5 },
            xaxis: { title: 'Lot size — thousands of sq ft (x₁)', gridcolor: 'lightgray', range: [1.5, 10.5] },
            yaxis: { title: 'Living area — sq ft (x₂)', gridcolor: 'lightgray', range: [400, 3200] },
            width: 780,
            height: 560,
            hovermode: 'closest',
            responsive: true,
            legend: { x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'lightgray', borderwidth: 1 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            margin: { t: 90 }
        }, { displayModeBar: false });
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
})();
