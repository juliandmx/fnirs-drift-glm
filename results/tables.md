# Ergebnistabellen (erzeugt von sweep_report.py)

## Beste Driftfamilie je Bedingung (minimaler RMSE_med)

| chromo | window_s | constellation | family | rmse_med | absbias_med | var_med |
| --- | --- | --- | --- | --- | --- | --- |
| HbO | 90 | baseline | butter:0.01 | 0.357 | 0.087 | 0.125 |
| HbO | 90 | global | poly:3 | 0.093 | 0.038 | 0.006 |
| HbO | 90 | motion | poly:1 | 0.364 | 0.088 | 0.127 |
| HbO | 90 | short_avg | dct:0.01 | 0.080 | 0.042 | 0.005 |
| HbO | 90 | short_maxcorr | poly:2 | 0.093 | 0.039 | 0.006 |
| HbO | 180 | baseline | poly:3 | 0.186 | 0.084 | 0.031 |
| HbO | 180 | global | dct:0.005 | 0.057 | 0.024 | 0.002 |
| HbO | 180 | motion | butter:0.01 | 0.185 | 0.079 | 0.033 |
| HbO | 180 | short_avg | butterxy:0.01 | 0.051 | 0.031 | 0.002 |
| HbO | 180 | short_maxcorr | butterxy:0.01 | 0.059 | 0.033 | 0.001 |
| HbO | 368 | baseline | butter:0.01 | 0.082 | 0.034 | 0.005 |
| HbO | 368 | global | dct:0.005 | 0.049 | 0.018 | 0.001 |
| HbO | 368 | motion | none | 0.080 | 0.030 | 0.005 |
| HbO | 368 | short_avg | none | 0.044 | 0.022 | 0.001 |
| HbO | 368 | short_maxcorr | dct:0.02 | 0.041 | 0.025 | 0.001 |
| HbR | 90 | baseline | legendre:3 | 0.054 | 0.020 | 0.002 |
| HbR | 90 | global | dct:0.005 | 0.069 | 0.032 | 0.004 |
| HbR | 90 | motion | poly:4 | 0.053 | 0.019 | 0.002 |
| HbR | 90 | short_avg | poly:2 | 0.093 | 0.032 | 0.008 |
| HbR | 90 | short_maxcorr | dct:0.02 | 0.072 | 0.012 | 0.004 |
| HbR | 180 | baseline | butter:0.01 | 0.030 | 0.016 | 0.000 |
| HbR | 180 | global | none | 0.041 | 0.028 | 0.001 |
| HbR | 180 | motion | butter:0.01 | 0.027 | 0.017 | 0.000 |
| HbR | 180 | short_avg | dct:0.005 | 0.075 | 0.047 | 0.003 |
| HbR | 180 | short_maxcorr | butterxy:0.01 | 0.042 | 0.027 | 0.001 |
| HbR | 368 | baseline | dct:0.005 | 0.030 | 0.011 | 0.000 |
| HbR | 368 | global | dct:0.02 | 0.024 | 0.009 | 0.000 |
| HbR | 368 | motion | legendre:3 | 0.027 | 0.009 | 0.000 |
| HbR | 368 | short_avg | dct:0.02 | 0.026 | 0.007 | 0.001 |
| HbR | 368 | short_maxcorr | poly:2 | 0.026 | 0.008 | 0.000 |

## RMSE_med je Familie × Fenster — HbO (baseline)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | 0.358 | 0.193 | 0.083 |
| poly:1 | 0.359 | 0.190 | 0.085 |
| poly:2 | 0.359 | 0.188 | 0.084 |
| poly:3 | 0.372 | 0.186 | 0.082 |
| poly:4 | 0.378 | 0.211 | 0.084 |
| poly:5 | 0.406 | 0.211 | 0.086 |
| dct:0.005 | 0.358 | 0.193 | 0.084 |
| dct:0.01 | 0.358 | 0.187 | 0.084 |
| dct:0.02 | 0.367 | 0.216 | 0.086 |
| legendre:1 | 0.359 | 0.190 | 0.085 |
| legendre:3 | 0.372 | 0.186 | 0.082 |
| legendre:5 | 0.406 | 0.211 | 0.086 |
| bspline:5 | 0.394 | 0.210 | 0.084 |
| bspline:8 | 0.489 | 0.211 | 0.085 |
| butter:0.01 | 0.357 | 0.189 | 0.082 |
| butterxy:0.01 | 0.358 | 0.191 | 0.083 |

## RMSE_med je Familie × Fenster — HbR (baseline)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | 0.062 | 0.044 | 0.031 |
| poly:1 | 0.061 | 0.041 | 0.030 |
| poly:2 | 0.061 | 0.036 | 0.030 |
| poly:3 | 0.054 | 0.036 | 0.030 |
| poly:4 | 0.058 | 0.035 | 0.032 |
| poly:5 | 0.060 | 0.042 | 0.032 |
| dct:0.005 | 0.062 | 0.044 | 0.030 |
| dct:0.01 | 0.062 | 0.039 | 0.032 |
| dct:0.02 | 0.060 | 0.043 | 0.034 |
| legendre:1 | 0.061 | 0.041 | 0.030 |
| legendre:3 | 0.054 | 0.036 | 0.030 |
| legendre:5 | 0.060 | 0.042 | 0.032 |
| bspline:5 | 0.062 | 0.036 | 0.032 |
| bspline:8 | 0.081 | 0.043 | 0.034 |
| butter:0.01 | 0.061 | 0.030 | 0.030 |
| butterxy:0.01 | 0.059 | 0.038 | 0.033 |

## Konstellations-Effekt (Mittel RMSE_med über Familien)

| chromo | constellation | rmse_med_mean |
| --- | --- | --- |
| HbO | baseline | 0.220 |
| HbO | global | 0.076 |
| HbO | motion | 0.219 |
| HbO | short_avg | 0.069 |
| HbO | short_maxcorr | 0.081 |
| HbR | baseline | 0.044 |
| HbR | global | 0.049 |
| HbR | motion | 0.044 |
| HbR | short_avg | 0.073 |
| HbR | short_maxcorr | 0.051 |

## Variance explained (adj. R²) je Familie × Fenster — HbO (baseline; Filter-Arme: R² auf der gefilterten Zeitreihe)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | -0.093 | -0.002 | 0.022 |
| poly:1 | -0.018 | 0.183 | 0.132 |
| poly:2 | -0.059 | 0.121 | 0.274 |
| poly:3 | -0.822 | 0.261 | 0.209 |
| poly:4 | -0.687 | 0.219 | 0.214 |
| poly:5 | -94.976 | -0.299 | 0.193 |
| dct:0.005 | -0.093 | -0.002 | 0.279 |
| dct:0.01 | -0.093 | 0.330 | 0.319 |
| dct:0.02 | 0.032 | 0.348 | 0.381 |
| legendre:1 | -0.018 | 0.183 | 0.132 |
| legendre:3 | -0.822 | 0.261 | 0.209 |
| legendre:5 | -94.976 | -0.299 | 0.193 |
| bspline:5 | -0.842 | 0.192 | 0.219 |
| bspline:8 | -6416.471 | -1.269 | 0.123 |
| butter:0.01 | -0.450 | 0.088 | 0.039 |
| butterxy:0.01 | -0.475 | 0.085 | 0.034 |

## Variance explained (adj. R²) je Familie × Fenster — HbR (baseline; Filter-Arme: R² auf der gefilterten Zeitreihe)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | -0.017 | -0.191 | -0.267 |
| poly:1 | 0.103 | 0.226 | 0.461 |
| poly:2 | 0.131 | 0.451 | 0.533 |
| poly:3 | 0.271 | 0.354 | 0.535 |
| poly:4 | 0.399 | 0.512 | 0.566 |
| poly:5 | 0.388 | 0.552 | 0.651 |
| dct:0.005 | -0.017 | -0.191 | 0.547 |
| dct:0.01 | -0.017 | 0.472 | 0.693 |
| dct:0.02 | 0.147 | 0.572 | 0.765 |
| legendre:1 | 0.103 | 0.226 | 0.461 |
| legendre:3 | 0.271 | 0.354 | 0.535 |
| legendre:5 | 0.388 | 0.552 | 0.651 |
| bspline:5 | 0.397 | 0.513 | 0.555 |
| bspline:8 | 0.361 | 0.604 | 0.689 |
| butter:0.01 | -0.015 | 0.074 | 0.224 |
| butterxy:0.01 | 0.008 | 0.134 | 0.220 |

## Residuen ↔ GT-Abweichung: corr(Residual-RMS, |β̂−GT|) über Seeds × Kanäle (Mittel über Fenster, baseline)

| chromo | family | resid_err_corr |
| --- | --- | --- |
| HbO | bspline:5 | 0.109 |
| HbO | bspline:8 | 0.060 |
| HbO | butter:0.01 | 0.165 |
| HbO | butterxy:0.01 | 0.250 |
| HbO | dct:0.005 | 0.233 |
| HbO | dct:0.01 | 0.260 |
| HbO | dct:0.02 | 0.157 |
| HbO | legendre:1 | 0.178 |
| HbO | legendre:3 | 0.143 |
| HbO | legendre:5 | 0.164 |
| HbO | none | 0.218 |
| HbO | poly:1 | 0.178 |
| HbO | poly:2 | 0.185 |
| HbO | poly:3 | 0.143 |
| HbO | poly:4 | 0.103 |
| HbO | poly:5 | 0.164 |
| HbR | bspline:5 | 0.183 |
| HbR | bspline:8 | 0.199 |
| HbR | butter:0.01 | 0.270 |
| HbR | butterxy:0.01 | 0.384 |
| HbR | dct:0.005 | -0.036 |
| HbR | dct:0.01 | 0.106 |
| HbR | dct:0.02 | 0.310 |
| HbR | legendre:1 | 0.123 |
| HbR | legendre:3 | 0.108 |
| HbR | legendre:5 | 0.178 |
| HbR | none | -0.050 |
| HbR | poly:1 | 0.123 |
| HbR | poly:2 | 0.104 |
| HbR | poly:3 | 0.108 |
| HbR | poly:4 | 0.196 |
| HbR | poly:5 | 0.178 |

## Flexible Recovery-Basis (Form-Treue)

| family | chromo | shape_corr_med | amp_rel_err_med | n |
| --- | --- | --- | --- | --- |
| bspline:5 | HbO | 0.784 | -0.045 | 80 |
| bspline:5 | HbR | 0.928 | 0.094 | 80 |
| butter:0.01 | HbO | 0.796 | -0.020 | 80 |
| butter:0.01 | HbR | 0.918 | 0.049 | 80 |
| dct:0.01 | HbO | 0.790 | -0.014 | 80 |
| dct:0.01 | HbR | 0.930 | 0.102 | 80 |
| legendre:3 | HbO | 0.791 | -0.017 | 80 |
| legendre:3 | HbR | 0.930 | 0.106 | 80 |
| none | HbO | 0.791 | -0.026 | 80 |
| none | HbR | 0.882 | -0.026 | 80 |
| poly:1 | HbO | 0.792 | -0.028 | 80 |
| poly:1 | HbR | 0.914 | 0.056 | 80 |
| poly:3 | HbO | 0.791 | -0.017 | 80 |
| poly:3 | HbR | 0.930 | 0.106 | 80 |
| poly:5 | HbO | 0.788 | -0.023 | 80 |
| poly:5 | HbR | 0.941 | 0.133 | 80 |

## Detektion nach FDR (q=0.05)

| family | chromo | sensitivity | specificity | precision | youden_J | n_detected | n_truth_active |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bspline:5 | HbO | 0.363 | 0.826 | 0.336 | 0.189 | 100.000 | 51.000 |
| bspline:5 | HbR | 0.692 | 0.902 | 0.273 | 0.594 | 66.500 | 26.000 |
| dct:0.01 | HbO | 0.353 | 0.835 | 0.289 | 0.188 | 95.000 | 51.000 |
| dct:0.01 | HbR | 0.654 | 0.929 | 0.329 | 0.583 | 52.000 | 26.000 |
| legendre:3 | HbO | 0.353 | 0.831 | 0.336 | 0.184 | 97.000 | 51.000 |
| legendre:3 | HbR | 0.673 | 0.913 | 0.292 | 0.586 | 60.500 | 26.000 |
| none | HbO | 0.373 | 0.844 | 0.253 | 0.217 | 92.000 | 51.000 |
| none | HbR | 0.442 | 0.947 | 0.306 | 0.390 | 37.500 | 26.000 |
| poly:1 | HbO | 0.363 | 0.851 | 0.321 | 0.214 | 88.000 | 51.000 |
| poly:1 | HbR | 0.635 | 0.888 | 0.248 | 0.523 | 71.500 | 26.000 |
| poly:3 | HbO | 0.353 | 0.831 | 0.336 | 0.184 | 97.000 | 51.000 |
| poly:3 | HbR | 0.673 | 0.913 | 0.292 | 0.586 | 60.500 | 26.000 |
