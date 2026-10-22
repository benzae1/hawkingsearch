Median final error over seeds 0-14, budget 10,000, d=10. W/T/L = hawking significantly better / no difference / worse (Mann-Whitney U, p<0.05).

| variant | tuned | sphere | rosenbrock | rastrigin | ackley | griewank | schwefel | terrain | iris-kmeans | wine-kmeans | mean rank | W/T/L vs hawking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hawking | alpha=0.001 | 0.00762 | 23.2 | 37.8 | 17.8 | 0.878 | 1.17e+03 | 4.17 | 0.00186 | 2.98 | 5.56 | - |
| hawking-dt1 | alpha=0.01 | 0.00854 | 21.6 | 42.8 | 18 | 0.88 | 1.17e+03 | 2.68 | 0.00162 | 2.91 | 5.11 | 0/8/1 |
| cool-earned | alpha=0.003 | 0.00929 | 23.5 | 36.2 | 16.9 | 0.871 | 1.1e+03 | 3.42 | 0.00198 | 2.97 | 5.00 | 1/8/0 |
| onefifth-earned | alpha=0.001 | 3.36e-07 | 3.34 | 27.9 | 2.04 | 0.0399 | 1.05e+03 | 4.41e-07 | 4.25e-06 | 2.08 | 1.44 | 0/1/8 |
| heat-fixedL | L=100 | 0.0113 | 37.2 | 33.9 | 13.5 | 0.874 | 1.06e+03 | 3.45 | 0.00217 | 2.82 | 4.78 | 1/7/1 |
| cool-fixedL | L=100 | 0.0116 | 19.4 | 38.6 | 16.4 | 0.899 | 1.07e+03 | 3.76 | 0.00196 | 2.99 | 5.67 | 0/8/1 |
| onefifth-L | L=100 | 3.36e-07 | 3.03 | 27.9 | 1.65 | 0.0399 | 1.05e+03 | 8.82e-09 | 4.15e-06 | 2.08 | 1.56 | 0/1/8 |
| fixed-L | L=300 | 0.479 | 134 | 50.6 | 7.2 | 2.64 | 1.58e+03 | 5.95 | 0.0353 | 2.32 | 6.89 | 6/1/2 |
