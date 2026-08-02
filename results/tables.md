# Ergebnistabellen (auto-generiert von sweep_report.py)

## Beste Driftfamilie je Bedingung (minimaler RMSE_med)

| chromo | window_s | constellation | family | rmse_med | absbias_med | var_med |
| --- | --- | --- | --- | --- | --- | --- |
| HbO | 90 | baseline | butter:0.01 | 0.475 | 0.111 | 0.202 |
| HbO | 90 | global | poly:2 | 0.133 | 0.083 | 0.016 |
| HbO | 90 | motion | butter:0.01 | 0.464 | 0.119 | 0.192 |
| HbO | 90 | short_avg | legendre:1 | 0.118 | 0.045 | 0.012 |
| HbO | 90 | short_maxcorr | butter:0.01 | 0.127 | 0.060 | 0.010 |
| HbO | 180 | baseline | poly:1 | 0.319 | 0.161 | 0.074 |
| HbO | 180 | global | legendre:1 | 0.110 | 0.041 | 0.008 |
| HbO | 180 | motion | bspline:8 | 0.324 | 0.169 | 0.068 |
| HbO | 180 | short_avg | legendre:1 | 0.124 | 0.070 | 0.006 |
| HbO | 180 | short_maxcorr | poly:1 | 0.092 | 0.073 | 0.005 |
| HbO | 368 | baseline | dct:0.01 | 0.119 | 0.040 | 0.008 |
| HbO | 368 | global | dct:0.01 | 0.053 | 0.034 | 0.003 |
| HbO | 368 | motion | butter:0.01 | 0.117 | 0.043 | 0.007 |
| HbO | 368 | short_avg | poly:1 | 0.083 | 0.068 | 0.002 |
| HbO | 368 | short_maxcorr | dct:0.01 | 0.057 | 0.034 | 0.002 |
| HbR | 90 | baseline | dct:0.02 | 0.091 | 0.027 | 0.007 |
| HbR | 90 | global | legendre:1 | 0.102 | 0.034 | 0.009 |
| HbR | 90 | motion | poly:2 | 0.097 | 0.031 | 0.007 |
| HbR | 90 | short_avg | poly:4 | 0.104 | 0.041 | 0.006 |
| HbR | 90 | short_maxcorr | butter:0.01 | 0.094 | 0.029 | 0.007 |
| HbR | 180 | baseline | bspline:8 | 0.056 | 0.027 | 0.002 |
| HbR | 180 | global | dct:0.02 | 0.075 | 0.022 | 0.003 |
| HbR | 180 | motion | butter:0.01 | 0.060 | 0.032 | 0.003 |
| HbR | 180 | short_avg | bspline:8 | 0.061 | 0.026 | 0.003 |
| HbR | 180 | short_maxcorr | bspline:8 | 0.061 | 0.029 | 0.003 |
| HbR | 368 | baseline | dct:0.02 | 0.025 | 0.016 | 0.001 |
| HbR | 368 | global | dct:0.02 | 0.034 | 0.015 | 0.001 |
| HbR | 368 | motion | butter:0.01 | 0.031 | 0.013 | 0.001 |
| HbR | 368 | short_avg | dct:0.01 | 0.030 | 0.013 | 0.001 |
| HbR | 368 | short_maxcorr | dct:0.02 | 0.022 | 0.009 | 0.000 |

## RMSE_med je Familie × Fenster — HbO (baseline)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | 0.493 | 0.320 | 0.125 |
| poly:1 | 0.494 | 0.319 | 0.124 |
| poly:2 | 0.490 | 0.343 | 0.124 |
| poly:3 | 0.506 | 0.345 | 0.124 |
| poly:4 | 0.515 | 0.345 | 0.123 |
| poly:5 | 0.518 | 0.343 | 0.123 |
| dct:0.005 | 0.493 | 0.320 | 0.120 |
| dct:0.01 | 0.493 | 0.345 | 0.119 |
| dct:0.02 | 0.496 | 0.332 | 0.121 |
| legendre:1 | 0.494 | 0.319 | 0.124 |
| legendre:3 | 0.506 | 0.345 | 0.124 |
| legendre:5 | 0.518 | 0.343 | 0.123 |
| bspline:5 | 0.518 | 0.345 | 0.123 |
| bspline:8 | 0.619 | 0.339 | 0.122 |
| butter:0.01 | 0.475 | 0.343 | 0.119 |

## RMSE_med je Familie × Fenster — HbR (baseline)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | 0.108 | 0.074 | 0.037 |
| poly:1 | 0.107 | 0.058 | 0.032 |
| poly:2 | 0.091 | 0.062 | 0.031 |
| poly:3 | 0.101 | 0.059 | 0.031 |
| poly:4 | 0.099 | 0.058 | 0.031 |
| poly:5 | 0.108 | 0.056 | 0.032 |
| dct:0.005 | 0.108 | 0.074 | 0.031 |
| dct:0.01 | 0.108 | 0.059 | 0.030 |
| dct:0.02 | 0.091 | 0.058 | 0.025 |
| legendre:1 | 0.107 | 0.058 | 0.032 |
| legendre:3 | 0.101 | 0.059 | 0.031 |
| legendre:5 | 0.108 | 0.056 | 0.032 |
| bspline:5 | 0.100 | 0.058 | 0.031 |
| bspline:8 | 0.128 | 0.056 | 0.031 |
| butter:0.01 | 0.102 | 0.056 | 0.029 |

## Konstellations-Effekt (Mittel RMSE_med über Familien)

| chromo | constellation | rmse_med_mean |
| --- | --- | --- |
| HbO | baseline | 0.323 |
| HbO | global | 0.117 |
| HbO | motion | 0.315 |
| HbO | short_avg | 0.126 |
| HbO | short_maxcorr | 0.116 |
| HbR | baseline | 0.065 |
| HbR | global | 0.075 |
| HbR | motion | 0.069 |
| HbR | short_avg | 0.072 |
| HbR | short_maxcorr | 0.070 |

## Flexible Recovery-Basis (Form-Treue)

| family | chromo | shape_corr_med | amp_rel_err_med | n |
| --- | --- | --- | --- | --- |
| bspline:5 | HbO | 0.874 | 0.400 | 80 |
| bspline:5 | HbR | 0.832 | -0.179 | 80 |
| butter:0.01 | HbO | 0.913 | 0.739 | 80 |
| butter:0.01 | HbR | 0.814 | -0.193 | 80 |
| dct:0.01 | HbO | 0.903 | 0.471 | 80 |
| dct:0.01 | HbR | 0.847 | -0.129 | 80 |
| legendre:3 | HbO | 0.885 | 0.416 | 80 |
| legendre:3 | HbR | 0.833 | -0.173 | 80 |
| none | HbO | 0.913 | 0.418 | 80 |
| none | HbR | 0.797 | -0.273 | 80 |
| poly:1 | HbO | 0.896 | 0.313 | 80 |
| poly:1 | HbR | 0.812 | -0.226 | 80 |
| poly:3 | HbO | 0.885 | 0.416 | 80 |
| poly:3 | HbR | 0.833 | -0.173 | 80 |
| poly:5 | HbO | 0.874 | 0.413 | 80 |
| poly:5 | HbR | 0.844 | -0.152 | 80 |

## Detektion nach FDR (q=0.05)

| family | chromo | sensitivity | specificity | precision | youden_J | n_detected | n_truth_active |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bspline:5 | HbO | 0.616 | 0.747 | 0.344 | 0.363 | 172.000 | 95.000 |
| bspline:5 | HbR | 0.698 | 0.816 | 0.296 | 0.514 | 127.500 | 53.000 |
| dct:0.01 | HbO | 0.595 | 0.775 | 0.368 | 0.370 | 157.500 | 95.000 |
| dct:0.01 | HbR | 0.670 | 0.848 | 0.329 | 0.518 | 110.000 | 53.000 |
| legendre:3 | HbO | 0.616 | 0.748 | 0.349 | 0.364 | 171.500 | 95.000 |
| legendre:3 | HbR | 0.670 | 0.820 | 0.291 | 0.490 | 124.000 | 53.000 |
| none | HbO | 0.526 | 0.804 | 0.362 | 0.330 | 138.000 | 95.000 |
| none | HbR | 0.528 | 0.891 | 0.341 | 0.419 | 81.500 | 53.000 |
| poly:1 | HbO | 0.505 | 0.793 | 0.341 | 0.298 | 141.000 | 95.000 |
| poly:1 | HbR | 0.500 | 0.844 | 0.258 | 0.344 | 103.000 | 53.000 |
| poly:3 | HbO | 0.616 | 0.748 | 0.349 | 0.364 | 171.500 | 95.000 |
| poly:3 | HbR | 0.670 | 0.820 | 0.291 | 0.490 | 124.000 | 53.000 |
