# Ablation: Hawking full with one module switched on or off

Median final error (f - f_opt) over 15 seeds; lower is better. **Bold** = best on that problem. W/T/L = default significantly better / no significant difference / worse (Mann-Whitney U, p < 0.05).

| algorithm | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 7.21e-23 | 0.299 | 8.95 | 2.48e-11 | 0.103 | 1.62e+03 | 0.000248 | 1.67e-15 | 0.664 | 6.89 |
| without gravity | 5.08e-24 | 0.374 | **7.96** | 2.58 | 0.162 | 1.56e+03 | 0.0865 | 9.99e-16 | 0.0115 | 8.44 |
| without gravitational_waves | **1.64e-25** | 0.584 | 9.95 | 2.58 | 0.364 | 1.62e+03 | **5.03e-08** | 3.29e-05 | 0.0101 | 12.11 |
| without area_theorem | 1.2e-22 | 0.413 | 8.95 | 2.01 | 0.171 | 1.62e+03 | 14.6 | 3.29e-05 | 0.000431 | 12.11 |
| without page_curve | 8.7e-23 | 0.264 | 8.95 | 2.48e-11 | 0.135 | 1.52e+03 | 6.39e-05 | **5.55e-16** | 0.664 | 6.33 |
| without spin | 1.09e-22 | 0.387 | 7.96 | 0.55 | 0.238 | 1.52e+03 | 0.0751 | 3.29e-05 | 7.1e-05 | 9.56 |
| without spaghettification | 8.77e-23 | 0.26 | 8.95 | 2.47e-11 | **0.0591** | 1.62e+03 | 36 | 2.66e-15 | 0.00468 | 8.22 |
| with slingshot | 1.01e-22 | 0.241 | 8.95 | 2.45e-10 | 0.15 | 1.56e+03 | 0.00154 | 3.29e-05 | 0.00186 | 8.00 |
| with white_holes | 1.11e-22 | 0.569 | 8.95 | 2.48e-11 | 0.0984 | 1.62e+03 | 0.0209 | 1.89e-15 | 0.664 | 9.89 |
| without penrose | 3.52e-05 | 0.847 | 8.95 | 1.62 | 0.299 | 1.52e+03 | 2.34 | 3.39e-05 | 0.00118 | 13.67 |
| without penrose_page_dirs | 8.96e-23 | 0.61 | 9.95 | **1.98e-11** | 0.187 | 1.56e+03 | 10.5 | 1.55e-15 | 0.00342 | 10.44 |
| without penrose_adaptive | 5.17e-11 | 0.406 | 8.95 | 1.47e-05 | 0.158 | 1.62e+03 | 0.265 | 1.29e-08 | 0.00468 | 12.00 |
| without endgame | 8.96e-16 | 0.568 | 7.96 | 0.125 | 0.109 | 1.62e+03 | 0.000248 | 8.04e-07 | 1.11 | 11.00 |
| without galactic_centre | 1.56e-23 | 0.309 | 37.8 | 1.65 | 0.148 | 1.3e+03 | 0.384 | 1.55e-15 | 0.0344 | 10.00 |
| without levy_radiation | 9.82e-23 | 0.164 | 9.95 | 2.39e-11 | 0.124 | 1.8e+03 | 0.00457 | 1.55e-15 | **9.8e-07** | 7.56 |
| without cmb_spawn | 1.5e-22 | **0.0841** | 11.9 | 4e-11 | 0.153 | 1.66e+03 | 0.0584 | 1.11e-15 | 0.00468 | 10.33 |
| without gw_adaptive | 1.37e-22 | 0.261 | 8.95 | 1.08 | 0.121 | 1.62e+03 | 0.0249 | 9.99e-16 | 1.34e-06 | 8.56 |
| every module on | 1.23e-22 | 0.428 | 8.95 | 2.45e-10 | 0.138 | 1.56e+03 | 0.000132 | 4.18e-06 | 0.00186 | 9.44 |
| no modules | 1.27e-05 | 7.91 | 45.1 | 18.5 | 0.147 | **1.15e+03** | 2.83 | 0.000199 | 2.41 | 15.44 |

## default vs each method (W/T/L across problems)

| opponent | W | T | L | problems where default wins |
|---|---|---|---|---|
| without gravity | 1 | 8 | 0 | ackley |
| without gravitational_waves | 2 | 7 | 0 | ackley, griewank |
| without area_theorem | 1 | 8 | 0 | ackley |
| without page_curve | 0 | 8 | 1 | - |
| without spin | 2 | 7 | 0 | ackley, griewank |
| without spaghettification | 1 | 8 | 0 | terrain |
| with slingshot | 0 | 9 | 0 | - |
| with white_holes | 0 | 9 | 0 | - |
| without penrose | 6 | 3 | 0 | sphere, rosenbrock, ackley, griewank, terrain, iris-kmeans |
| without penrose_page_dirs | 2 | 7 | 0 | rosenbrock, griewank |
| without penrose_adaptive | 1 | 8 | 0 | ackley |
| without endgame | 1 | 8 | 0 | ackley |
| without galactic_centre | 2 | 6 | 1 | rastrigin, ackley |
| without levy_radiation | 0 | 9 | 0 | - |
| without cmb_spawn | 0 | 9 | 0 | - |
| without gw_adaptive | 1 | 8 | 0 | ackley |
| every module on | 0 | 9 | 0 | - |
| no modules | 7 | 1 | 1 | sphere, rosenbrock, rastrigin, ackley, terrain, iris-kmeans, wine-kmeans |
