# Practical Problems

This repository provides real-world optimization problems as
[IOHexperimenter](https://iohprofiler.github.io/) (`ioh`) problems. You can use them alongside
IOH's built-in benchmark suites, log experiments with IOH, and analyze the results with
[IOHanalyzer](https://iohanalyzer.liacs.nl/).

## Installation

Create and activate a virtual environment, then install IOH and this package:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install "ioh<=0.3.22"
python -m pip install git+https://github.com/BayesOptApp/Practical_Problems.git
```

Alternatively, clone the repository and install the local checkout:

```bash
git clone https://github.com/BayesOptApp/Practical_Problems.git
cd Practical_Problems
python -m pip install .
```

The package installation also installs its declared dependencies (`ioh<=0.3.22` and
`FLORIS==2.5.2`). The `bench_fun_eps` and `bench_fun_pitz` problems additionally need Docker or
Podman. WindWake plotting can be enabled with the optional `plot` dependency:
`python -m pip install ".[plot]"`.

## Available practical problems

| ID   | Problem                              | Allowed dimensions |
|------|--------------------------------------|--------------------|
| 1121 | Bench_Fun_Eps                        | 49                 |
| 1122 | Bench_Fun_Pitz                       | 10                 |
| 1123 | Circular_Antenna_Array                | 12                 |
| 1124 | Frequency_Modulated_Sound_Waves       | 6, 12, 24, 48      |
| 1125 | Lennard_Jones_Potential               | 6, 12, 24, 48      |
| 1126 | Spacecraft_Trajectory_OptimizationC1  | 26                 |
| 1127 | Spacecraft_Trajectory_OptimizationC2  | 22                 |
| 1128 | Spread_Spectrum_Radar                 | 20                 |
| 1129 | Tersoff_PotentialC1                   | 6, 12, 24, 48      |
| 1130 | WindWake                              | 6, 12, 24, 48      |

Use one of the listed dimensions; unsupported dimensions raise `AttributeError`. All practical
problems accept optimizer inputs in `[-5, 5]^d`, which are mapped internally to each problem's
native bounds.

## Creating and evaluating problems

IOH provides `get_problem` for its built-in BBOB suite. For example, create and evaluate the
5-dimensional Sphere problem (BBOB function 1, instance 1):

```python
import ioh

bbob_problem = ioh.get_problem(
    1,
    dimension=5,
    instance=1,
    problem_class=ioh.ProblemClass.BBOB,
)
value = bbob_problem([0.0] * 5)
print(bbob_problem.meta_data.name, value)
```

Create a practical problem with `get_practical_problem(id, dim=...)`:

```python
from Practical_Problems import get_practical_problem

practical_problem = get_practical_problem(1126, dim=26)
value = practical_problem([0.0] * 26)
print(practical_problem.meta_data.name, value)
```

Problem-specific keyword arguments are available for `WindWake` (`wind_seed`, `n_samples`,
`plotting`) and the two benchmark-function problems (`precision`, `call_mode`, where `call_mode`
is `"docker"` or `"podman"`). See [`Example_Python_Implementation.ipynb`](Example_Python_Implementation.ipynb)
for a longer practical-problem example.

## Logging and running experiments

IOH loggers attach to a problem object, whether it came from BBOB's `get_problem` or this
package's `get_practical_problem`. The same experiment loop can therefore be used with either:

```python
import os
import numpy as np
import ioh
from Practical_Problems import get_practical_problem

# Choose either problem constructor:
problem = ioh.get_problem(
    1, dimension=5, instance=1, problem_class=ioh.ProblemClass.BBOB
)
# Or use a practical problem instead:
# problem = get_practical_problem(1126, dim=26)

logger = ioh.logger.Analyzer(
    triggers=[ioh.logger.trigger.OnImprovement(), ioh.logger.trigger.Each(10)],
    root=os.path.curdir,
    algorithm_name="RandomSearch",
    folder_name="Results-RS",
    algorithm_info="Uniform random search",
    store_positions=False,
)
problem.attach_logger(logger)

rng = np.random.default_rng(42)
dimension = problem.meta_data.n_variables
for run in range(3):
    best = float("inf")
    for _ in range(1000):
        candidate = rng.uniform(-5, 5, dimension)
        best = min(best, problem(candidate))
    print(f"Run {run + 1}: best objective = {best}")
    problem.reset()  # Start a fresh logged run.

problem.detach_logger()  # Flush and close the logger.
```

The example samples uniformly from the problems' `[-5, 5]` search domain. Logger output is written
to `Results-RS/` and can be uploaded to IOHanalyzer. To log another problem, create it, attach a
logger, run the experiment, and detach the logger when finished.
