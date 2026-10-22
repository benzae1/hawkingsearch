# Endgame ringdown validation

Median final error (f - f_opt) over 15 seeds; lower is better. **Bold** = best on that problem. W/T/L = late-ringdown significantly better / no significant difference / worse (Mann-Whitney U, p < 0.05).

| algorithm | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 1e-22 | 0.491 | 7.96 | 2.37e-11 | 0.0911 | 1.56e+03 | 1.66e-06 | 3.55e-15 | 4.82e-09 | 4.00 |
| ringdown | 3.16e-30 | 5.2e-15 | 8.95 | 3.37e-11 | **0** | 1.48e+03 | 2.81e-05 | **2.22e-16** | 4.17e-10 | 2.89 |
| late-ringdown | 3.16e-30 | 4.86e-15 | 7.96 | 3.59e-13 | 0.0911 | 1.56e+03 | **1.11e-06** | **2.22e-16** | **0** | 2.56 |
| late-ringdown+quanta | **2.37e-30** | **4.85e-15** | 6.96 | **3.02e-13** | 0.0394 | 1.37e+03 | 1.71 | **2.22e-16** | **0** | 2.22 |
| dual-annealing (scipy) | 1.05e-13 | 1.44e-10 | **2.84e-14** | 1.75e-08 | 0.0246 | **0.000127** | 0.000119 | 2.78e-11 | 4.95e-11 | 3.33 |

## late-ringdown vs each method (W/T/L across problems)

| opponent | W | T | L | problems where late-ringdown wins |
|---|---|---|---|---|
| original | 2 | 7 | 0 | rosenbrock, wine-kmeans |
| ringdown | 2 | 5 | 2 | ackley, wine-kmeans |
| late-ringdown+quanta | 1 | 7 | 1 | terrain |
| dual-annealing (scipy) | 3 | 3 | 3 | rosenbrock, iris-kmeans, wine-kmeans |
