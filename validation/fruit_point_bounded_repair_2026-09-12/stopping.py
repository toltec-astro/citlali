"""Same positive starlet objective and L-BFGS-B path; explicit KKT completion."""
import time
import numpy as np
from scipy.optimize import minimize
from common import candidate

OPERATIONAL_TOL = 1e-4
TIGHT_TOL = 1e-6
OPERATIONAL_ITERATIONS = 3000
OPERATIONAL_EVALUATIONS = 30000
TIGHT_ITERATIONS = 12000
TIGHT_EVALUATIONS = 120000

def completion(v, fun, grad, gradient_scale):
    finite = bool(np.isfinite(v).all() and np.isfinite(fun) and np.isfinite(grad).all()
                  and np.isfinite(gradient_scale) and gradient_scale > 0)
    feasible = bool(finite and np.all(v >= 0))
    pg = np.where((v <= 0) & (grad > 0), 0., grad)
    relative = float(np.max(np.abs(pg))/gradient_scale) if finite else float('inf')
    return dict(finite=finite, feasible=feasible, relative_projected_gradient=relative,
                objective=float(fun), minimum_variable=float(np.min(v)))

def problem(e, z):
    Y, beta = e.residual(z)
    W = candidate.analysis(Y.reshape(e.shape)).reshape(5, -1)
    sigma = e.sigmas[:, e.stratum]
    omega = e.valid & e.D & (abs(W) >= 5*sigma)
    outer = candidate.mad(Y[e.O])
    scale = e.normalization_fallback if outer == 0 else outer
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('normalization unavailable')
    weights = np.where(omega, (scale/sigma)**2, 0).reshape((5,)+e.shape)
    targets = candidate.analysis((Y/scale).reshape(e.shape))
    def objective(v):
        canvas = np.zeros_like(Y)
        canvas[e.D] = v
        diff = candidate.analysis(canvas.reshape(e.shape))-targets
        return .5*float(np.sum(weights*diff*diff)), candidate.adjoint(weights*diff).ravel()[e.D]
    zero = np.zeros(int(e.D.sum()))
    _, g0 = objective(zero)
    return dict(objective=objective, zero=zero, scale=scale, background=beta,
                omega=omega, gradient_scale=max(float(np.max(abs(g0))), 1e-12))

class Finished(Exception):
    pass

def solve_path(p, *, qualify=True, seconds_limit=180):
    """Capture both stopping points without restarting the limited-memory path."""
    start = time.monotonic()
    snapshots = {}
    trace = []
    iterations = 0
    evaluations = 0
    cache = {}
    def objective(v):
        nonlocal evaluations
        evaluations += 1
        f, g = p['objective'](v)
        cache.update(v=v.copy(), f=f, g=g.copy())
        return f, g
    def inspect(v):
        if not cache or not np.array_equal(cache['v'], v):
            objective(v)
        check = completion(v, cache['f'], cache['g'], p['gradient_scale'])
        return dict(**check, iterations=iterations, function_evaluations=evaluations,
                    seconds=time.monotonic()-start)
    def callback(v):
        nonlocal iterations
        iterations += 1
        check = inspect(v)
        if iterations == 1 or iterations % 100 == 0:
            trace.append(check)
        if check['finite'] and check['feasible']:
            for label, tolerance in [('declared', OPERATIONAL_TOL), ('tight', TIGHT_TOL)]:
                if label not in snapshots and check['relative_projected_gradient'] <= tolerance:
                    snapshots[label] = dict(check, v=v.copy())
        if ('tight' in snapshots if qualify else 'declared' in snapshots):
            raise Finished('declared_numerical_criterion')
        if time.monotonic()-start > seconds_limit:
            raise Finished('saved_problem_time_limit')
    options = dict(maxiter=TIGHT_ITERATIONS if qualify else OPERATIONAL_ITERATIONS,
                   maxfun=TIGHT_EVALUATIONS if qualify else OPERATIONAL_EVALUATIONS,
                   ftol=0., gtol=0., maxcor=10, maxls=20)
    try:
        result = minimize(objective, p['zero'], jac=True, method='L-BFGS-B',
                          bounds=[(0, None)]*len(p['zero']), callback=callback, options=options)
        message = str(result.message)
        last = result.x.copy()
    except Finished as err:
        message = str(err)
        last = cache['v'].copy()
    final = dict(inspect(last), v=last)
    declared = snapshots.get('declared')
    available = bool(declared is not None and declared['iterations'] <= OPERATIONAL_ITERATIONS
                     and declared['function_evaluations'] <= OPERATIONAL_EVALUATIONS)
    return dict(snapshots=snapshots, final=final, trace=trace, termination=message,
                operational_stop_available=available, wall_seconds=time.monotonic()-start,
                solver_options=options)
