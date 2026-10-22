# Optional refinement study

Median final error (f - f_opt) over 5 seeds; lower is better. **Bold** = best on that problem. W/T/L = quanta significantly better / no significant difference / worse (Mann-Whitney U, p < 0.05).

| algorithm | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|
| quanta | 1.77e-23 | 0.208 | 8.95 | **3.1e-11** | 0.0812 | **1.22e+03** | **3.62e-07** | 1.55e-15 | 5.86e-08 | 1.67 |
| ringdown+quanta | **3.16e-30** | **5.95e-15** | **8.95** | 0.123 | **0.00986** | 1.38e+03 | 4.02 | **2.22e-16** | **6.17e-10** | 1.33 |

## quanta vs each method (W/T/L across problems)

| opponent | W | T | L | problems where quanta wins |
|---|---|---|---|---|
| ringdown+quanta | 0 | 7 | 2 | - |
