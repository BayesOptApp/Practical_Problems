# Practical_Problems
This repository includes a suite of practical (real-world) optimization problems, exposed as
[IOHexperimenter]([https://iohprofiler.github.io/](https://iohprofiler.github.io/IOHexperimenter/)) (`ioh`) problems so they can be logged and analysed with IOHprofiler.

## Installation
To install IOHexperimenter (`ioh`) on its own:
```bash
pip install "ioh<=0.3.22"
```

Installing this repository also installs `ioh` and the other required dependencies:
```bash
pip install .   # or: pip install git+https://github.com/BayesOptApp/Practical_Problems.git
```
(`ioh<=0.3.22`, `FLORIS`). The `bench_fun_eps` and `bench_fun_pitz` problems additionally require Docker or Podman.

## Available problems

| ID   | Problem                            | Allowed dimensions |
|------|------------------------------------|--------------------|
| 1121 | Bench_Fun_Eps                      | 49 |
| 1122 | Bench_Fun_Pitz                     | 10 |
| 1123 | Circular_Antenna_Array             | 12 |
| 1124 | Frequency_Modulated_Sound_Waves    | 6, 12, 24, 48 |
| 1125 | Lennard_Jones_Potential            | 6, 12, 24, 48 |
| 1126 | Spacecraft_Trajectory_OptimizationC1 | 26 |
| 1127 | Spacecraft_Trajectory_OptimizationC2 | 22 |
| 1128 | Spread_Spectrum_Radar              | 20 |
| 1129 | Tersoff_PotentialC1                | 6, 12, 24, 48 |
| 1130 | WindWake                           | 6, 12, 24, 48 |

Requesting a dimensionality that is not allowed for a problem raises an `AttributeError`.

## Search range
Every problem is exposed to the optimizer in the box **[-5, 5]^d** (as in BBOB). Internally, each
coordinate is linearly mapped to the real bounds of the practical problem (which differ per variable
and per problem), so your optimizer only ever needs to sample in `[-5, 5]`.

## Example
A full version is in [`Example_Python_Implementation.ipynb`](Example_Python_Implementation.ipynb).

### 1. Create the problem
Use `get_practical_problem(id, dim=...)`; pick a dimension valid for that problem (see table above).
```python
import os
import numpy as np
import ioh
from Practical_Problems import get_practical_problem

problem = get_practical_problem(1126, dim=26)   # Spacecraft_Trajectory_OptimizationC1
print(problem.meta_data.name, problem.meta_data.n_variables)
```
Extra keyword arguments: `wind_seed`, `n_samples`, `plotting` (WindWake), and `precision`, `call_mode`
(`"docker"`/`"podman"`, for the bench_fun problems).

### 2. Create an IOHprofiler logger and attach it
```python
logger = ioh.logger.Analyzer(
    triggers=[ioh.logger.trigger.OnImprovement(), ioh.logger.trigger.Each(10)],
    root=os.path.curdir,
    algorithm_name="RandomSearch",
    folder_name="Results-RS",
    algorithm_info="A simple random search optimizer",
    store_positions=False,
)
problem.attach_logger(logger)
```

### 3. Run an optimizer in [-5, 5]
```python
rng = np.random.default_rng(42)
best = float("inf")
for _ in range(1000):
    x = rng.uniform(-5, 5, problem.meta_data.n_variables)
    best = min(best, problem(x))
print(best)

problem.reset()          # call between independent runs so the logger starts a new run
problem.detach_logger()  # flush and close the logger when finished
```
Results are written to `Results-RS/` and can be uploaded to the
[IOHanalyzer](https://iohanalyzer.liacs.nl/) for analysis.

To run several problems/dimensions with the same logger, loop over them and call
`attach_logger` / `detach_logger` for each one.
