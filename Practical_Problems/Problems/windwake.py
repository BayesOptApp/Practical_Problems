"""Version-dispatched FLORIS wind layout benchmark.

FLORIS 2.5.2: original implementation, including original constraint behavior.
FLORIS 4.3 <= version < 5: modern implementation (tested with 4.7), corrected
physical spacing/boundary checks. Values are not identical across backends.
Place windwake_input.json and windwake_input_v4.yaml beside this module.
The v2 backend still requires its own compatible NumPy environment.
"""
import os
from pathlib import Path
import numpy as np
from packaging.version import Version

try:
    import floris
except ModuleNotFoundError as exc:
    if exc.name != 'floris':
        raise
    raise ModuleNotFoundError('Install floris==2.5.2 or floris>=4.3,<5') from exc

FLORIS_IMPORTED_VERSION = str(floris.__version__)
_version = Version(FLORIS_IMPORTED_VERSION)
_DEFAULT_DIRECTORY = Path(__file__).resolve().parent

if _version == Version('2.5.2'):
    import floris.tools as wfct
elif Version('4.3') <= _version < Version('5'):
    from floris import FlorisModel, TimeSeries
else:
    raise ImportError(
        f'Unsupported FLORIS version {FLORIS_IMPORTED_VERSION}. '
        'Supported: 2.5.2 or >=4.3,<5 (v4 backend tested with 4.7).'
    )

class WindWakeLayoutV2:

    def __init__(self, 
                file=None, 
                n_turbines:int=3, 
                wind_seed:int=0, 
                width=None, 
                height=None, 
                n_samples:int=5,
                plotting:bool=False):

        file = _DEFAULT_DIRECTORY / 'windwake_input.json' if file is None else file
        self.file = file
        self.wind_seed = wind_seed
        self.n_turbines = n_turbines

        self.width = width if width is not None else 333.33 * n_turbines
        self.height = height if height is not None else 333.33 * n_turbines
        # How many times to sample the matrix. Default is to use a fixed sample.
        self.n_samples = n_samples
        #self.wind_rng = np.random.RandomState(wind_seed)
        self.wind_rng:np.random.Generator = np.random.default_rng(seed=wind_seed)
        self.wd, self.ws, self.freq = self._gen_random_wind()
        # self.loggerclass = logging.getLoggerClass()
        self.fi = wfct.floris_interface.FlorisInterface(self.file)
        # Set number of turbines
        #rand_layout_x = np.random.uniform(0.0, self.width, size=n_turbines)
        #rand_layout_y = np.random.uniform(0.0, self.height, size=n_turbines)
        rand_layout_x = self.wind_rng.uniform(0.0, self.width, size=n_turbines)
        rand_layout_y = self.wind_rng.uniform(0.0, self.height, size=n_turbines)

        self.fi.reinitialize_flow_field(layout_array=(rand_layout_x, rand_layout_y))

        # Default polygon (covers entire area)
        self.boundaries = [[0.0, 0.0], [self.width, 0.0], [self.width, self.height], [0.0, self.height]]
        # Scaling factor, set to 1 in order to avoid scaling.
        self.aep_initial = 1
        self.lo = wfct.optimization.scipy.layout.LayoutOptimization(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
        # Use the default minimum distance that floris themselves use.
        self.min_dist = self.lo.min_dist
        self.plotting = plotting
        # logging.setLoggerClass(self.loggerclass)

    def _gen_random_wind(self):
        rng = self.wind_rng
        wd = np.arange(0.0, 360.0, 5.0)
        ws = 8.0 + rng.standard_normal(len(wd)) * 0.5
        freq = np.abs(np.sort(rng.standard_normal(len(wd))))
        freq = freq / freq.sum()
        return wd, ws, freq

    def evaluate(self, x):
        c1 = self.lo._space_constraint(x, self.min_dist)
        c2 = self.lo._distance_from_boundaries(x, self.boundaries)

        # No power produced when constraints are violated.
        if c1 < 0 or c2 < 0:
            return 0.0

        if self.n_samples is None:
            obj = self.lo._AEP_layout_opt(x)
        else:
            obj = 0.0
            for _ in range(self.n_samples):
                # Resample wind speed
                self.ws = 8.0 + self.wind_rng.standard_normal(size=len(self.wd,)) * 0.5
                self.lo = wfct.optimization.scipy.layout.LayoutOptimization(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
                obj += self.lo._AEP_layout_opt(x)
            obj = obj / self.n_samples

        #if self.plotting:

            #fig, ax = self.plot_layout_opt_results(x)

        dictt = self.fi.get_model_parameters()
            #plt.show()


        return obj

    def plot_layout_opt_results(self,x:np.ndarray)->object:
        """
        This method plots the locations of the turbines in a
        wind farm after layout optimization.
        """

        import matplotlib.pyplot as plt

        locsx = [
            self.lo._unnorm(valx, self.lo.bndx_min, self.lo.bndx_max)
            for valx in x[0 : self.lo.nturbs]
        ]
        locsy = [
            self.lo._unnorm(valy, self.lo.bndy_min, self.lo.bndy_max)
            for valy in x[self.lo.nturbs : 2 * self.lo.nturbs]
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        fontsize = 16
        ax.plot(locsx, locsy, "or")
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        ax.set_xlabel("x (m)", fontsize=fontsize)
        ax.set_ylabel("y (m)", fontsize=fontsize)
        ax.axis("equal")
        ax.grid()
        ax.tick_params(which="both", labelsize=fontsize)
        ax.legend(
            ["Old locations", "New locations"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts) - 1:
                ax.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
            else:
                ax.plot(
                    [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
                )
        return fig, ax

    def lbs(self):
        return np.zeros(2 * self.n_turbines, dtype=float)

    def ubs(self):
        # return (np.asarray([[self.width, self.height]]) * np.ones((self.n_turbines, 1), dtype=float)).ravel()
        return np.ones(2 * self.n_turbines, dtype=float)

    def vartype(self):
        return np.array(['cont'] * self.dims())

    def dims(self):
        return self.n_turbines * 2

    def __str__(self):
        return f"WindWakeLayout(file={self.file},n_turbines={self.n_turbines},width={self.width},height={self.height},wind_seed={self.wind_seed})"


class WindWakeLayoutV4:
    def __init__(self, file=None, n_turbines=3, wind_seed=0, width=None,
                 height=None, n_samples=5, plotting=False, *,
                 turbulence_intensity=0.06, min_dist=None):
        if not isinstance(n_turbines, (int, np.integer)) or n_turbines < 1:
            raise ValueError('n_turbines must be a positive integer')
        if n_samples is not None and (
            not isinstance(n_samples, (int, np.integer)) or n_samples < 1
        ):
            raise ValueError('n_samples must be None or a positive integer')
        file = _DEFAULT_DIRECTORY / 'windwake_input_v4.yaml' if file is None else file
        self.file, self.n_turbines, self.wind_seed = file, n_turbines, wind_seed
        self.width = float(333.33 * n_turbines if width is None else width)
        self.height = float(333.33 * n_turbines if height is None else height)
        if not np.all(np.isfinite([self.width, self.height])) or min(self.width, self.height) <= 0:
            raise ValueError('width and height must be finite and positive')
        self.n_samples, self.plotting = n_samples, plotting


        
        self.wind_rng = np.random.default_rng(seed=wind_seed)
        self.wd, self.ws, self.freq = self._gen_random_wind()

        self.turbulence_intensity = np.asarray([float(turbulence_intensity)] * len(self.wd))

        
        if isinstance(file, (str, Path)) and Path(file).suffix.lower() == '.json':
            raise ValueError('Migrate the FLORIS v2 JSON to a v4 YAML/dict first')
        
        self.fi = FlorisModel('defaults' if file is None else file)
        # Preserve the old random-number consumption before the first evaluation.
        lx = self.wind_rng.uniform(0, self.width, n_turbines)
        ly = self.wind_rng.uniform(0, self.height, n_turbines)
        types = self.fi.core.farm.turbine_type

        if len(types) not in (1, n_turbines):
            raise ValueError('Configuration must contain one turbine type or one per turbine')
        
        self.fi.set(layout_x=lx, layout_y=ly)
        self.min_dist = float(2 * self.fi.core.farm.rotor_diameters[0]
                              if min_dist is None else min_dist)
        if not np.isfinite(self.min_dist) or self.min_dist < 0:
            raise ValueError('min_dist must be finite and nonnegative')
        self.boundaries = [[0., 0.], [self.width, 0.],
                           [self.width, self.height], [0., self.height]]
        self.aep_initial = 1.0

    def _gen_random_wind(self):
        wd = np.arange(0., 360., 5.)
        ws = 8. + self.wind_rng.standard_normal(len(wd)) * 0.5
        freq = np.abs(np.sort(self.wind_rng.standard_normal(len(wd))))
        return wd, ws, freq / freq.sum()

    def _coordinates(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dims(),):
            raise ValueError(f'x must have shape ({self.dims()},)')
        return x, np.column_stack((x[:self.n_turbines] * self.width,
                                   x[self.n_turbines:] * self.height))

    def evaluate(self, x):
        x, xy = self._coordinates(x)
        if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
            return 0.0
        i, j = np.triu_indices(self.n_turbines, k=1)
        if np.any(np.linalg.norm(xy[i] - xy[j], axis=1) < self.min_dist):
            return 0.0
        total = 0.0
        for _ in range(1 if self.n_samples is None else self.n_samples):
            if self.n_samples is not None:
                self.ws = 8. + self.wind_rng.standard_normal(len(self.wd)) * 0.5
            self.fi.set(
                layout_x=xy[:, 0], layout_y=xy[:, 1],
                wind_data=TimeSeries(wind_directions=self.wd, wind_speeds=self.ws,
                                     turbulence_intensities=self.turbulence_intensity),
            )
            self.fi.run()
            total -= float(self.fi.get_farm_AEP(freq=self.freq)) / self.aep_initial
        return total / (1 if self.n_samples is None else self.n_samples)

    def plot_layout_opt_results(self, x):
        import matplotlib.pyplot as plt
        _, xy = self._coordinates(x)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xy[:, 0], xy[:, 1], 'or')
        polygon = np.asarray(self.boundaries + [self.boundaries[0]])
        ax.plot(polygon[:, 0], polygon[:, 1], 'b-')
        ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal')
        ax.grid()
        return fig, ax

    def lbs(self):
        return np.zeros(self.dims())

    def ubs(self):
        return np.ones(self.dims())

    def vartype(self):
        return np.array(['cont'] * self.dims())

    def dims(self):
        return 2 * self.n_turbines

    def __str__(self):
        return (f'WindWakeLayout(file={self.file},n_turbines={self.n_turbines},'
                f'width={self.width},height={self.height},wind_seed={self.wind_seed})')

if _version == Version('2.5.2'):
    WindWakeLayout = WindWakeLayoutV2
else:
    WindWakeLayout = WindWakeLayoutV4


if __name__ == '__main__':
    problem = WindWakeLayout(n_turbines=3, wind_seed=0, n_samples=5)
    x = np.array([1/6, 0.831910785957425, 0.5, 1/6, 5/6, 0.5])
    print('FLORIS version:', FLORIS_IMPORTED_VERSION)
    print('Backend:', WindWakeLayout.__name__)
    print('Objective value:', problem.evaluate(x))
