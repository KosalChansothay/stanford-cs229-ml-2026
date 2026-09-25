/* ==========================================================================
   CS231N Lecture 1 Interactive Plotly Chart
   Model Capacity, Dataset Scale & Benchmark Accuracy Co-Evolution (1958-2023)
   Loaded by courses/cs231n/lecture-1.html AFTER markdown is rendered.
   ========================================================================== */

(function () {
    'use strict';

    function drawCapacityDataAccuracyChart(divId) {
        var el = document.getElementById(divId);
        if (!el) return;

        // 1. Classical / Hand-Engineered Systems
        var classical = {
            name: 'Perceptron (1958)',
            x: [100, 10000, 100000, 500000],
            y: [100, 1000, 10000, 100000],
            labels: [
                'Perceptron (1958)',
                'Neocognitron (1980)',
                'SIFT + SVM (2006)',
                'HOG + DPM (2010)'
            ],
            accuracies: [35.0, 52.0, 71.5, 73.8],
            details: [
                'First hardware neural perceptron<br>Single-layer linear threshold',
                'Hand-crafted receptive fields<br>No automated backprop learning',
                'Hand-engineered invariant features<br>PASCAL VOC benchmark',
                'Deformable part models<br>ILSVRC 2010 baseline (~28.2% error)'
            ],
            mode: 'markers+text',
            type: 'scatter',
            name: 'Classical / Hand-Designed',
            text: ['Perceptron', 'Neocognitron', 'SIFT+SVM', 'HOG+DPM'],
            textposition: 'top left',
            textfont: { size: 11, color: '#286b82' },
            marker: {
                symbol: 'triangle-up',
                size: [14, 16, 18, 20],
                color: '#286b82',
                line: { width: 1.5, color: '#173f5f' }
            },
            hovertemplate: '<b>%{text}</b><br>Parameters: %{x:~s}<br>Dataset Size: %{y:~s}<br>Top-5 Accuracy: %{customdata[0]}%<br><i>%{customdata[1]}</i><extra></extra>',
            customdata: [
                [35.0, 'Hardware perceptron (Single-layer)'],
                [52.0, 'Fukushima hand-designed visual hierarchy'],
                [71.5, 'Hand-engineered SIFT descriptors'],
                [73.8, 'ILSVRC 2010 classical baseline']
            ]
        };

        // 2. Deep Convolutional Networks (Backprop + GPUs + ImageNet)
        var deepCNN = {
            x: [60000, 60000000, 138000000, 25600000, 89000000],
            y: [60000, 1281167, 1281167, 1281167, 1281167],
            labels: [
                'LeNet-5 (1998)',
                'AlexNet (2012)',
                'VGG-16 (2014)',
                'ResNet-50 (2015)',
                'ConvNeXt-L (2022)'
            ],
            accuracies: [88.0, 84.7, 92.7, 96.4, 98.2],
            mode: 'markers+text',
            type: 'scatter',
            name: 'Deep Convolutional Networks',
            text: ['LeNet-5 (1998)', 'AlexNet (2012)', 'VGG-16 (2014)', 'ResNet-50 (2015)', 'ConvNeXt-L (2022)'],
            textposition: 'top right',
            textfont: { size: 11, color: '#d97706' },
            marker: {
                symbol: 'circle',
                size: [16, 22, 26, 30, 32],
                color: '#d97706',
                line: { width: 2, color: '#92400e' }
            },
            hovertemplate: '<b>%{text}</b><br>Parameters: %{x:~s}<br>Dataset Size: %{y:~s}<br>Top-5 Accuracy: %{customdata[0]}%<br><i>%{customdata[1]}</i><extra></extra>',
            customdata: [
                [88.0, 'Yann LeCun backprop for digit recognition'],
                [84.7, 'The 2012 DL rebirth: 8 layers, 2x GPUs, ImageNet'],
                [92.7, 'Homogeneous 3x3 conv stacks (Oxford VGG)'],
                [96.4, 'Skip connections (He et al.) surpassing human accuracy (94.9%)'],
                [98.2, 'Modernized Pure-CNN baseline competing with ViT']
            ]
        };

        // 3. Vision Transformers & Foundation Models
        var transformers = {
            x: [86000000, 304000000, 1100000000],
            y: [14000000, 400000000, 142000000],
            labels: [
                'ViT-Base (2020)',
                'CLIP ViT-L (2021)',
                'DINOv2-g (2023)'
            ],
            accuracies: [98.1, 98.7, 99.2],
            mode: 'markers+text',
            type: 'scatter',
            name: 'Vision Transformers & Foundation Models',
            text: ['ViT-Base (2020)', 'CLIP ViT-L (2021)', 'DINOv2-g (2023)'],
            textposition: 'bottom right',
            textfont: { size: 11, color: '#2563eb' },
            marker: {
                symbol: 'diamond',
                size: [24, 28, 34],
                color: '#2563eb',
                line: { width: 2, color: '#1e40af' }
            },
            hovertemplate: '<b>%{text}</b><br>Parameters: %{x:~s}<br>Dataset Size: %{y:~s}<br>Top-5 Accuracy: %{customdata[0]}%<br><i>%{customdata[1]}</i><extra></extra>',
            customdata: [
                [98.1, 'Patch projection + self-attention (Dosovitskiy et al.)'],
                [98.7, 'Contrastive Vision-Language pre-training on 400M images'],
                [99.2, 'Self-supervised visual representations on 142M images']
            ]
        };

        var data = [classical, deepCNN, transformers];

        var layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            title: {
                text: '<b>Model Capacity, Dataset Scale & Benchmark Accuracy Co-Evolution (1958–2023)</b>',
                x: 0.5,
                y: 0.98,
                font: { size: 15 }
            },
            xaxis: {
                title: 'Model Parameters (Log Scale)',
                type: 'log',
                tickmode: 'array',
                tickvals: [100, 10000, 1000000, 100000000, 1000000000],
                ticktext: ['100', '10K', '1M', '100M', '1B'],
                gridcolor: '#e5e5e5'
            },
            yaxis: {
                title: 'Training Dataset Size (Images, Log Scale)',
                type: 'log',
                tickmode: 'array',
                tickvals: [100, 10000, 1000000, 10000000, 100000000, 500000000],
                ticktext: ['100', '10K', '1M', '10M', '100M', '500M'],
                gridcolor: '#e5e5e5'
            },
            legend: {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: 1.15,
                font: { size: 11 }
            },
            annotations: [
                {
                    x: Math.log10(60000000),
                    y: Math.log10(1281167),
                    text: '<b>2012: AlexNet Rebirth</b><br>GPU compute + ImageNet',
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 1.5,
                    arrowcolor: '#d97706',
                    ax: -50,
                    ay: -45,
                    font: { size: 10, color: '#92400e' },
                    bgcolor: 'rgba(255, 251, 235, 0.85)',
                    bordercolor: '#d97706',
                    borderwidth: 1
                },
                {
                    x: Math.log10(25600000),
                    y: Math.log10(1281167),
                    text: '<b>2015: ResNet Superhuman</b><br>3.57% Top-5 Error (< 5.1% Human)',
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 1.5,
                    arrowcolor: '#d97706',
                    ax: 55,
                    ay: 45,
                    font: { size: 10, color: '#92400e' },
                    bgcolor: 'rgba(255, 251, 235, 0.85)',
                    bordercolor: '#d97706',
                    borderwidth: 1
                }
            ],
            updatemenus: [
                {
                    type: 'buttons',
                    direction: 'right',
                    x: 0.5,
                    xanchor: 'center',
                    y: -0.22,
                    buttons: [
                        {
                            label: 'All Paradigms',
                            method: 'restyle',
                            args: ['visible', [true, true, true]]
                        },
                        {
                            label: 'Deep Learning Eras (CNN & ViT)',
                            method: 'restyle',
                            args: ['visible', [false, true, true]]
                        },
                        {
                            label: 'Classical vs Early CNN',
                            method: 'restyle',
                            args: ['visible', [true, true, false]]
                        }
                    ]
                }
            ],
            height: 540,
            responsive: true,
            hovermode: 'closest',
            margin: { t: 95, b: 85, l: 75, r: 50 }
        };

        var config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(el, data, layout, config);
    }

    window.renderCs231nLecture1Charts = function () {
        drawCapacityDataAccuracyChart('plotly-cs231n-1-blueprint');
    };

    document.addEventListener('markdown:rendered', window.renderCs231nLecture1Charts);
    if (document.readyState !== 'loading' && document.getElementById('plotly-cs231n-1-blueprint')) {
        window.renderCs231nLecture1Charts();
    }
})();
