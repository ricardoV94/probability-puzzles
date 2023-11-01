import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import matplotlib.pyplot as plt

from pytensor.graph import ancestors
from pytensor.tensor.random.op import RandomVariable
try:
    from pymc.model.core import BlockModelAccess
except AttributeError:
    from pymc.model import BlockModelAccess
    
# These are missing from pm.math
pm.math.arcsinh = pt.arcsinh
pm.math.cumsum = pt.cumsum
pm.math.max = pt.max
pm.math.round = pt.round
pm.math.floor = pt.floor

def check_valid_graph(rv):
    allowed_rvs = (
        pm.Normal,
        pm.Gamma,
        pm.Beta,
        pm.StudentT,
        pm.MvNormal,
    )
    
    for var in ancestors([rv]):
        if var.owner and isinstance(var.owner.op, RandomVariable):
            if not isinstance(var.owner.op, allowed_rvs):
                raise ValueError(
                    f"RV {var.owner.op.name.capitalize()} is not allowed!\n"
                    f"Answer can contain only {', '.join([rv.rv_op.name.capitalize() for rv in allowed_rvs])} "
                    "and operations from the math module"
                )
    
    
def get_inputs_and_solution(exercise):

    if exercise == "Log-Normal":
        params = dict(
            mu=-np.log(2),
            sigma=np.log(2),
        )        
        ref = pm.LogNormal.dist(**params)
        test_values = np.linspace(-1, 3, 100)
        
    elif exercise == "Exp-Normal":
        params = dict(
            mu=np.e*2,
            sigma=np.log(2),
        )        
        ref = pm.math.log(pm.Normal.dist(**params))
        test_values = np.linspace(-1, 3, 100)
    
    elif exercise == "Half-Normal":
        params = dict(
            sigma=np.log(2),
        )        
        ref = pm.HalfNormal.dist(**params)
        test_values = np.linspace(-3, 3, 100)
        
    elif exercise == "Shifted-Half-Normal":
        params = dict(
            sigma=np.log(2),
            shift=-np.e/2,
        )        
        ref = pm.HalfNormal.dist(params["sigma"]) + params["shift"]
        test_values = np.linspace(-3, 3, 100)
    
    elif exercise == "Negative-Half-Normal":
        params = dict(
            sigma=np.log(2),
        )        
        ref = -pm.HalfNormal.dist(**params)
        test_values = np.linspace(-3, 3, 100)
        
    elif exercise == "Reciprocal-Normal":
        params = dict(
            mu=0.05,
            sigma=np.e*1.5,
        )        
        ref = 1 / pm.Normal.dist(**params)
        test_values = np.linspace(-3, 3, 100)
        
    elif exercise == "Logit-Normal":
        params = dict(
            mu=np.log(2),
            sigma=np.pi,
        )        
        ref = pm.LogitNormal.dist(**params)
        test_values = np.linspace(-0.5, 1.5, 100)
        
    elif exercise == "Inverse-Gamma":
        params = dict(
            alpha=np.pi,
            beta=1+1/11,
        )        
        ref = pm.InverseGamma.dist(**params)
        test_values = np.linspace(-0.5, 3.0, 100)
        
    elif exercise == "Log-Gamma":
        params = dict(
            alpha=np.pi,
            beta=1+1/11,
        )        
        ref = pm.math.log(pm.Gamma.dist(**params))
        test_values = np.linspace(-2.0, 3.0, 100)
        
    elif exercise == "Generalized-Gamma":
        params = dict(
            alpha=np.pi * 1.5,
            d=1+1/11,
            p=np.e*2,
        )  
        params = dict(
            alpha=np.pi * 1.5,
            d=1 + 1/101,
            p=np.pi * 1.5,
        ) 
        q = 1/params["p"]
        k = params["d"] * q
        theta = params["alpha"] ** (1 / q)
        ref = pm.Gamma.dist(k, 1/theta) ** q
        test_values = np.linspace(-0.5, 8.0, 100)
        
    elif exercise == "Chi2-one-dof":
        params = dict()
        ref = pm.ChiSquared.dist(nu=1)
        test_values = np.linspace(-1, 3, 100)
        
    elif exercise == "Chi2-one-dof-mean":
        params = dict(
            mean=np.sqrt(np.sqrt(2)),
        )
        ref = pm.ChiSquared.dist(nu=1) + params["mean"] ** 2
        test_values = np.linspace(-1, 3, 100)
        
    elif exercise == "Chi2-n-dof":
        params = dict(dof=np.pi)
        ref = pm.ChiSquared.dist(nu=params["dof"])
        test_values = np.linspace(-1, 7, 100)
        
    elif exercise == "Chi-n-dof":
        params = dict(dof=np.pi)
        ref = pm.math.sqrt(pm.ChiSquared.dist(nu=params["dof"]))
        test_values = np.linspace(-1, 7, 100)
        
    elif exercise == "F-one-dof1-n-dof2":
        params = dict(dof2=np.pi+1)
        ref = pm.math.sqr(pm.StudentT.dist(nu=params["dof2"], mu=0, sigma=1))
        test_values = np.linspace(-1, 3, 100)
        
    elif exercise == "F-n-dof1-one-dof2":
        params = dict(dof1=np.pi+1)
        ref = 1/(pm.StudentT.dist(nu=params["dof1"], mu=0, sigma=1)) ** 2
        test_values = np.linspace(-1, 3, 100)
    
    elif exercise == "Reciprocal":
        params = dict(
            a=np.log(2),
            b=np.pi,
        )
        ref = pm.math.exp(pm.Uniform.dist(np.log(params["a"]), np.log(params["b"])))
        test_values = np.linspace(0, 4, 100)
    
    elif exercise == "Exponential":
        params = dict(lam=1/11,)
        ref = pm.Exponential.dist(**params)
        test_values = np.linspace(-1, 30, 100)
        
    elif exercise == "Uniform":
        params = dict()
        ref = pm.Uniform.dist()
        test_values = np.linspace(-0.5, 1.5, 100)
        
    elif exercise == "Weibull":
        params = dict(
            lam=np.log(2),
            k=np.e,
        )
        ref = pm.Weibull.dist(beta=params["lam"], alpha=params["k"])
        test_values = np.linspace(-1, 3, 100)
        
    elif exercise == "PERT":
        params = dict(
            min=np.pi,
            max=np.pi*3,
            m=np.pi * 1.5,
            lam=3.4,
        )
        
        high = params["max"]
        low = params["min"]
        peak = params["m"]
        lmbda = params["lam"]
        range_ = (high - low)
        alpha = 1 + lmbda * (peak - low) / range_
        beta = 1 + lmbda * (high - peak) / range_
        
        ref = pm.Beta.dist(alpha, beta) * range_ + low
        test_values = np.linspace(-1, 11, 100)
            
    elif exercise == "Maxwell-Boltzmann":
        params = dict(scale=np.pi)    
        
        ref = pm.math.sqrt(pm.ChiSquared.dist(nu=3)) * params["scale"]
        test_values = np.linspace(-1, 12, 100)
        
        
    elif exercise == "Sinh-Arcsinh-Normal":
        params = dict(
            mu=np.log(2),
            sigma=np.sqrt(2),
            nu=-np.pi/2,
            tau=1 + 1/11,
        )    
        
        mu = params["mu"]
        sigma = params["sigma"]
        nu = params["nu"]
        tau = params["tau"]
        
        ref = mu + sigma * pm.math.sinh((pm.math.arcsinh(pm.Normal.dist()) + nu) / tau)
        test_values = np.linspace(-6, 6, 100)
        
    elif exercise == "Multivariate-Log-Normal":
        params = dict(
            mu=[np.pi/2, np.pi/2],
            cov=np.eye(2) * np.log(2.0),
        )
        ref = pm.math.exp(pm.MvNormal.dist(**params))
        
        X, Y = np.meshgrid(np.linspace(0.1, 40, 75), np.linspace(0.1, 40, 75))
        test_values = np.dstack([X, Y]).reshape(-1, 2)
        
    elif exercise == "Normal-Random-Walk":
        params = dict(
            x0=np.pi,
            drift=-2/11,
            sigma=np.log(2),
            n_steps=50,
        )
        ref = (
            params["x0"]
            + pm.Normal.dist(
                mu=params["drift"], 
                sigma=params["sigma"], 
                shape=(params["n_steps"]),
            ).cumsum()
        )
        
        rng = np.random.default_rng(123)
        test_values = rng.normal(size=(1, params["n_steps"]))
        
    elif exercise == "Maximum-N-IID-Normal":
        params = dict(
            mu=np.log(2),
            sigma=np.log(2),
            n=3,
        )
        ref = pm.math.max(pm.Normal.dist(params["mu"], params["sigma"], shape=params["n"]))
        test_values = np.linspace(-1, 3, 100)
    
    elif exercise == "Censored-Normal":
        params = dict(mu=np.log(2), sigma=np.sqrt(2), lower=0, upper=np.pi)
        ref = pm.Censored.dist(
            pm.Normal.dist(params["mu"], params["sigma"]), 
            lower=params["lower"],
            upper=params["upper"],
        )
        test_values = np.linspace(params["lower"]-1, params["upper"]+1, 100)
        test_values = np.insert(test_values, 0, params["lower"])
        test_values = np.insert(test_values, 0, params["upper"])
    
    elif exercise == "Right-Censored-Normal":
        params = dict(mu=np.log(2), sigma=np.sqrt(2), upper=np.pi)
        ref = pm.Censored.dist(
            pm.Normal.dist(params["mu"], params["sigma"]), 
            lower=None,
            upper=params["upper"],
        )
        test_values = np.linspace(-3, params["upper"]+1, 100)
        test_values = np.insert(test_values, 0, params["upper"])
        
    elif exercise == "Discretized-Normal":
        params = dict(mu=np.log(2), sigma=np.sqrt(2))
        ref = pm.math.round(pm.Normal.dist(**params))
        test_values = np.arange(-10, 10+1).astype(ref.type.dtype)
        
    elif exercise == "Geometric":
        params = dict(p=1/3)
        ref = pm.Geometric.dist(**params)
        test_values = np.arange(1, 12+1).astype("float64")
         
    elif exercise == "Normal-Normal-Mixture":
        params = dict(w=[0.3, 0.7], mu=[-1.1, 1.1], sigma=np.log(2))
        ref = pm.NormalMixture.dist(**params)
        test_values = np.linspace(-3, 3, 100)
        
    else:
        raise ValueError(f"Unknown exercise: {exercise}")
        
    tensor_params = {k: pt.as_tensor(v, name=k) for k, v in params.items()}
    return ref, tensor_params, test_values
    

def plot_pdf(exercise, test_values, ref_logp_eval, rv_logp_eval=None):
    ylabel = "Probability function"
    
    if exercise == "Multivariate-Log-Normal":
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for axi in ax:
            axi.set_box_aspect(1)
            
        n = int(np.sqrt(ref_logp_eval.size))
        x = test_values[..., 0].reshape(n, n)
        y = test_values[..., 1].reshape(n, n)
        
        ax[0].contour(x, y, ref_logp_eval.reshape(n, n), levels=15)
        if rv_logp_eval is not None:
            ax[1].contour(x, y, rv_logp_eval.reshape(n, n), levels=15)
        
        ax[0].set_title("expected")
        ax[1].set_title("evaluated")        
        fig.suptitle(exercise)
        
    elif exercise == "Normal-Random-Walk":
        steps = np.arange(ref_logp_eval.size)
        plt.scatter(steps, np.exp(ref_logp_eval[0]), label="expected", color="k")
        if rv_logp_eval is not None:
            plt.scatter(steps, np.exp(rv_logp_eval[0]), label="evaluated", color="C0")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(exercise)
        plt.legend()
        
    elif "Censored" in exercise:
        y_max = np.max(np.exp(ref_logp_eval))
        n = 1 if "Right" in exercise else 2  # Number of censoring bounds
        censored_test_values, test_values = np.split(test_values, [n])
        censored_ref_logp_eval, ref_logp_eval = np.split(ref_logp_eval, [n])
        
        plt.plot(test_values, np.exp(ref_logp_eval), label="expected", color="k")
        plt.vlines(censored_test_values, -0.05, y_max+.1, color="w", lw=7)
        plt.scatter(censored_test_values, np.exp(censored_ref_logp_eval), color="k", zorder=5)
        if rv_logp_eval is not None:
            censored_rv_logp_eval, rv_logp_eval = np.split(rv_logp_eval, [n])
            plt.plot(test_values, np.exp(rv_logp_eval), label="evaluated", color="C0")
            plt.vlines(censored_test_values, -0.05, y_max+.1, color="w", lw=7)
            plt.scatter(censored_test_values, np.exp(censored_rv_logp_eval), color="C0", zorder=5)
        plt.ylabel(ylabel)
        plt.title(exercise)
        plt.legend()
        
    elif exercise in ("Discretized-Normal", "Geometric"):  # Discrete
        plt.scatter(test_values, np.exp(ref_logp_eval), label="expected", color="k")
        if rv_logp_eval is not None:
            plt.scatter(test_values, np.exp(rv_logp_eval), label="evaluated", color="C0")
        plt.ylabel(ylabel)
        plt.title(exercise)
        plt.legend()
        
    else:
        plt.plot(test_values, np.exp(ref_logp_eval), label="expected", color="k")
        if rv_logp_eval is not None:
            plt.plot(test_values, np.exp(rv_logp_eval), label="evaluated", color="C0")
        plt.ylabel(ylabel)
        plt.title(exercise)
        plt.legend()

def test(dist, exercise):
    
    ref, params, test_values = get_inputs_and_solution(exercise)
    
    msg = (
        "You are not allowed to use named variables or a PyMC Model.\n"
        "The answer should only contain variables created via the .dist() API "
        "and operations from the math module."
    )
    with BlockModelAccess(error_msg_on_access=msg):
        rv = dist(**params)
        
    if rv is ...:
        value = pt.tensor(shape=ref.type.shape, dtype=test_values.dtype)
        ref_logp = pm.logp(ref, value)
        ref_logp_fn = pytensor.function([value], ref_logp, mode="FAST_COMPILE")
        with np.errstate(all="ignore"):
            ref_logp_eval = np.array([ref_logp_fn(test_value) for test_value in test_values])
            
        print("Status: Incomplete")
        plot_pdf(exercise, test_values, ref_logp_eval, None)
        
    else:
        check_valid_graph(rv)
        
        value = pt.tensor(shape=ref.type.shape, dtype=test_values.dtype)
        rv_logp = pm.logp(rv, value)
        ref_logp = pm.logp(ref, value)
        
        rv_logp_fn = pytensor.function([value], rv_logp, mode="FAST_COMPILE")
        ref_logp_fn = pytensor.function([value], ref_logp, mode="FAST_COMPILE")
    
        with np.errstate(all="ignore"):
            rv_logp_eval = np.array([rv_logp_fn(test_value) for test_value in test_values])
            ref_logp_eval = np.array([ref_logp_fn(test_value) for test_value in test_values])

        try:
            np.testing.assert_allclose(
                rv_logp_eval, 
                ref_logp_eval, 
                rtol=1e-5,
                atol=1e-5,
            )
        except AssertionError:
            print("Status: ❌ Incorrect!")
        else:
            print("Status: ✅ Correct!")

        plot_pdf(exercise, test_values, ref_logp_eval, rv_logp_eval)
