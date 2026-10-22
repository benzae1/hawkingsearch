# Optional refinement study

Median final error (f - f_opt) over 15 seeds; lower is better. **Bold** = best on that problem. W/T/L = ringdown significantly better / no significant difference / worse (Mann-Whitney U, p < 0.05).

| algorithm | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 9.53e-23 | 0.292 | 7.96 | **2.78e-11** | 0.147 | 1.72e+03 | 3.74 | 1.19e-06 | 0.00468 | 5.22 |
| ringdown | 3.94e-30 | **5.11e-15** | 8.95 | 0.437 | 0.0296 | 1.68e+03 | 2.51e-05 | **2.22e-16** | 7.73e-09 | 3.44 |
| local-centre | 2.44e-23 | 0.632 | 21.9 | 0.0022 | 0.064 | 1.13e+03 | 2.93e-05 | 8.88e-16 | 1.71 | 5.11 |
| ringdown+local-centre | 3.94e-30 | 6.15e-15 | 31.8 | 1.16 | 0.027 | 1.17e+03 | 1.95e-05 | **2.22e-16** | 3.89e-08 | 3.78 |
| quanta | 2.99e-23 | 0.312 | 10.9 | 3.95e-11 | 0.0369 | 1.48e+03 | 1.99e-06 | 2.11e-15 | 0.0114 | 4.44 |
| ringdown+quanta | **3.16e-30** | 6.03e-15 | 7.96 | 0.0739 | **0.0074** | 1.41e+03 | **4.24e-07** | **2.22e-16** | 5.23e-09 | 2.44 |
| dual-annealing (scipy) | 2.55e-16 | 2.31e-10 | **2.84e-14** | 1.81e-08 | 0.0566 | **0.000127** | 2.38e-05 | 3.83e-11 | **3.67e-10** | 3.56 |

## ringdown vs each method (W/T/L across problems)

| opponent | W | T | L | problems where ringdown wins |
|---|---|---|---|---|
| original | 3 | 5 | 1 | rosenbrock, griewank, iris-kmeans |
| local-centre | 4 | 4 | 1 | rosenbrock, rastrigin, iris-kmeans, wine-kmeans |
| ringdown+local-centre | 2 | 6 | 1 | rosenbrock, rastrigin |
| quanta | 3 | 6 | 0 | rosenbrock, iris-kmeans, wine-kmeans |
| ringdown+quanta | 0 | 8 | 1 | - |
| dual-annealing (scipy) | 2 | 5 | 2 | rosenbrock, iris-kmeans |
