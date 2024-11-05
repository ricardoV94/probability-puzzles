from puzzle.src import test

# Add some extra imports missing from pymc.math
import numpy as np
import pytensor.tensor as pt
import pymc

pymc.math.inf = pymc.math.infty = pymc.math.Inf = pymc.math.Infinity = pymc.math.PINF = np.inf
pymc.math.ninf = pymc.math.NINF = -np.inf
pymc.math.pow = pymc.math.power = pt.pow
pymc.math.power = pt.pow
pymc.math.diag = pt.diag
pymc.math.linalg = pt.linalg
pymc.math.cholesky = pt.linalg.cholesky
pymc.math.eig = pt.linalg.eig
