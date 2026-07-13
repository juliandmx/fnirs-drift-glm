# Ergebnistabellen (auto-generiert von sweep_report.py)

## Beste Driftfamilie je Bedingung (minimaler RMSE_med)

| chromo | window_s | constellation | family | rmse_med | absbias_med | var_med |
| --- | --- | --- | --- | --- | --- | --- |
| HbO | 90 | baseline | butter:0.01 | 0.493 | 0.228 | 0.100 |
| HbO | 90 | global | legendre:3 | 0.187 | 0.109 | 0.013 |
| HbO | 90 | motion | butter:0.01 | 0.455 | 0.149 | 0.119 |
| HbO | 90 | motion+global | legendre:3 | 0.174 | 0.088 | 0.012 |
| HbO | 180 | baseline | legendre:1 | 0.223 | 0.174 | 0.023 |
| HbO | 180 | global | poly:1 | 0.147 | 0.126 | 0.004 |
| HbO | 180 | motion | poly:1 | 0.263 | 0.177 | 0.035 |
| HbO | 180 | motion+global | legendre:1 | 0.150 | 0.122 | 0.004 |
| HbO | 368 | baseline | dct:0.02 | 0.116 | 0.063 | 0.010 |
| HbO | 368 | global | butter:0.01 | 0.079 | 0.035 | 0.002 |
| HbO | 368 | motion | dct:0.02 | 0.125 | 0.067 | 0.010 |
| HbO | 368 | motion+global | butter:0.01 | 0.075 | 0.046 | 0.003 |
| HbR | 90 | baseline | legendre:3 | 0.066 | 0.046 | 0.002 |
| HbR | 90 | global | poly:4 | 0.073 | 0.038 | 0.005 |
| HbR | 90 | motion | dct:0.02 | 0.076 | 0.042 | 0.003 |
| HbR | 90 | motion+global | poly:2 | 0.075 | 0.041 | 0.003 |
| HbR | 180 | baseline | poly:1 | 0.063 | 0.029 | 0.003 |
| HbR | 180 | global | butter:0.01 | 0.050 | 0.028 | 0.002 |
| HbR | 180 | motion | legendre:1 | 0.063 | 0.022 | 0.003 |
| HbR | 180 | motion+global | butter:0.01 | 0.051 | 0.027 | 0.002 |
| HbR | 368 | baseline | dct:0.02 | 0.029 | 0.017 | 0.000 |
| HbR | 368 | global | dct:0.02 | 0.032 | 0.024 | 0.000 |
| HbR | 368 | motion | dct:0.02 | 0.029 | 0.017 | 0.001 |
| HbR | 368 | motion+global | dct:0.01 | 0.032 | 0.022 | 0.001 |

## RMSE_med je Familie × Fenster — HbO (baseline)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | 0.545 | 0.234 | 0.145 |
| poly:1 | 0.536 | 0.223 | 0.120 |
| poly:2 | 0.509 | 0.274 | 0.121 |
| poly:3 | 0.525 | 0.279 | 0.121 |
| poly:4 | 0.630 | 0.286 | 0.122 |
| poly:5 | 0.603 | 0.287 | 0.123 |
| dct:0.005 | 0.545 | 0.234 | 0.121 |
| dct:0.01 | 0.545 | 0.312 | 0.120 |
| dct:0.02 | 0.524 | 0.284 | 0.116 |
| legendre:1 | 0.536 | 0.223 | 0.120 |
| legendre:3 | 0.525 | 0.279 | 0.121 |
| legendre:5 | 0.603 | 0.287 | 0.123 |
| bspline:5 | 0.642 | 0.282 | 0.122 |
| bspline:8 | 0.690 | 0.295 | 0.126 |
| butter:0.01 | 0.493 | 0.307 | 0.140 |

## RMSE_med je Familie × Fenster — HbR (baseline)

| family | 90s | 180s | 368s |
| --- | --- | --- | --- |
| none | 0.096 | 0.067 | 0.035 |
| poly:1 | 0.080 | 0.063 | 0.033 |
| poly:2 | 0.074 | 0.064 | 0.035 |
| poly:3 | 0.066 | 0.064 | 0.035 |
| poly:4 | 0.073 | 0.064 | 0.035 |
| poly:5 | 0.074 | 0.065 | 0.035 |
| dct:0.005 | 0.096 | 0.067 | 0.033 |
| dct:0.01 | 0.096 | 0.064 | 0.033 |
| dct:0.02 | 0.069 | 0.064 | 0.029 |
| legendre:1 | 0.080 | 0.063 | 0.033 |
| legendre:3 | 0.066 | 0.064 | 0.035 |
| legendre:5 | 0.074 | 0.065 | 0.035 |
| bspline:5 | 0.078 | 0.064 | 0.035 |
| bspline:8 | 0.129 | 0.064 | 0.035 |
| butter:0.01 | 0.087 | 0.064 | 0.034 |

## Konstellations-Effekt (Mittel RMSE_med über Familien)

| chromo | constellation | rmse_med_mean |
| --- | --- | --- |
| HbO | baseline | 0.320 |
| HbO | global | 0.182 |
| HbO | motion | 0.317 |
| HbO | motion+global | 0.181 |
| HbR | baseline | 0.060 |
| HbR | global | 0.059 |
| HbR | motion | 0.063 |
| HbR | motion+global | 0.060 |

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
