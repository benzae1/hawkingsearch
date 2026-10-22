# Hawking search vs standard optimisers

Median final error (f - f_opt) over 15 seeds; lower is better. **Bold** = best on that problem. W/T/L = hawking-full significantly better / no significant difference / worse (Mann-Whitney U, p < 0.05).

| algorithm | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|
| hawking-minimal | 1.73e-05 | 35.5 | 37.4 | 16.4 | 0.245 | 1.03e+03 | 0.313 | 0.00402 | 2.9 | 6.11 |
| hawking-full | **7.21e-23** | 0.299 | 8.95 | 2.48e-11 | 0.103 | 1.62e+03 | 0.000248 | 1.67e-15 | 0.664 | 3.56 |
| random-search | 12.8 | 1.17e+04 | 69 | 16.9 | 45.1 | 1.84e+03 | 29.5 | 0.695 | 10.1 | 8.89 |
| simulated-annealing | 0.0529 | 5.32e+03 | 43.8 | 0.105 | 1.11 | 1.46e+03 | 8.72 | 0.104 | 5.96 | 7.56 |
| (1+1)-ES | 4.56e-16 | 1.58 | 33.8 | 4.42 | 0.0566 | 1.09e+03 | 1.08e-06 | **1.55e-15** | 1.73 | 4.22 |
| PSO | 9.52e-17 | 4.12 | 7.96 | 1.42e-07 | 0.0566 | 711 | **0** | 0.0123 | 1.71 | 3.22 |
| CMA-ES | 1.97e-13 | 2.66e-08 | 6.96 | **5.38e-12** | **1.84e-13** | 1.85e+03 | 4.3e-11 | 0.0123 | 5.73 | 3.89 |
| DE (scipy) | 2.03e-05 | 7.26 | 30.3 | 0.0516 | 0.75 | 712 | 1.49e-10 | 0.00026 | 5.22 | 5.11 |
| dual-annealing (scipy) | 2.96e-13 | **1.16e-10** | **2.84e-14** | 1.8e-08 | 0.0246 | **0.000127** | 8.91e-06 | 5.89e-11 | **4.28e-10** | 2.44 |

## hawking-full vs each method (W/T/L across problems)

| opponent | W | T | L | problems where hawking-full wins |
|---|---|---|---|---|
| hawking-minimal | 8 | 0 | 1 | sphere, rosenbrock, rastrigin, ackley, griewank, terrain, iris-kmeans, wine-kmeans |
| random-search | 9 | 0 | 0 | sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, terrain, iris-kmeans, wine-kmeans |
| simulated-annealing | 8 | 1 | 0 | sphere, rosenbrock, rastrigin, ackley, griewank, terrain, iris-kmeans, wine-kmeans |
| (1+1)-ES | 3 | 5 | 1 | rosenbrock, rastrigin, ackley |
| PSO | 3 | 4 | 2 | rosenbrock, ackley, iris-kmeans |
| CMA-ES | 2 | 4 | 3 | iris-kmeans, wine-kmeans |
| DE (scipy) | 7 | 0 | 2 | sphere, rosenbrock, rastrigin, ackley, griewank, iris-kmeans, wine-kmeans |
| dual-annealing (scipy) | 1 | 3 | 5 | ackley |
