# Optional refinement study

Median final error (f - f_opt) over 5 seeds; lower is better. **Bold** = best on that problem. W/T/L = ringdown significantly better / no significant difference / worse (Mann-Whitney U, p < 0.05).

| algorithm | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 5.37e-23 | 0.206 | 11.9 | 0.155 | 0.115 | 1.76e+03 | 3e-06 | 2.44e-15 | 1.75 | 4.00 |
| ringdown | 3.94e-30 | **5.08e-15** | 13.9 | 3.72e-11 | **0.0154** | 1.64e+03 | 0.000364 | **2.22e-16** | **6.49e-12** | 2.22 |
| local-centre | 2.59e-23 | 0.124 | 43.8 | **1.94e-11** | 0.0739 | 1.19e+03 | **1.97e-08** | 1.33e-15 | 1.82 | 3.22 |
| ringdown+local-centre | **2.37e-30** | 6.11e-15 | 41.8 | 1.79 | 0.0344 | 1.18e+03 | 5.64e-07 | **2.22e-16** | 0.00468 | 2.56 |
| dual-annealing (scipy) | 1.32e-13 | 3.21e-11 | **2.84e-14** | 1.78e-08 | 0.0394 | **0.000127** | 6.33e-06 | 3.03e-12 | 1.76e-11 | 3.00 |

## ringdown vs each method (W/T/L across problems)

| opponent | W | T | L | problems where ringdown wins |
|---|---|---|---|---|
| original | 3 | 6 | 0 | rosenbrock, griewank, iris-kmeans |
| local-centre | 3 | 6 | 0 | rosenbrock, iris-kmeans, wine-kmeans |
| ringdown+local-centre | 1 | 8 | 0 | rastrigin |
| dual-annealing (scipy) | 1 | 6 | 2 | iris-kmeans |
