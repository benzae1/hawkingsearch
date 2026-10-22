# Optional refinement study

Median final error (f - f_opt) over 5 seeds; lower is better. **Bold** = best on that problem. W/T/L = late-ringdown significantly better / no significant difference / worse (Mann-Whitney U, p < 0.05).

| algorithm | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|
| late-ringdown | 3.94e-30 | **4.9e-15** | 11.9 | 0.155 | 0.115 | 1.76e+03 | **4.64e-07** | **2.22e-16** | 1.74 | 1.67 |
| late-ringdown+quanta | **3.16e-30** | 4.91e-15 | **8.95** | **2.89e-14** | **0.0467** | **1.22e+03** | 6.99e-07 | **2.22e-16** | **0** | 1.33 |

## late-ringdown vs each method (W/T/L across problems)

| opponent | W | T | L | problems where late-ringdown wins |
|---|---|---|---|---|
| late-ringdown+quanta | 0 | 9 | 0 | - |
