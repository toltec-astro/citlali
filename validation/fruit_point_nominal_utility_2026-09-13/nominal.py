"""Adaptor for the already tested nominal stopping point; no estimator revision."""
import time
import numpy as np
from common import candidate, module, REPAIR
stopping = module('utility_frozen_stopping', REPAIR/'stopping.py')

class Estimator(candidate.Estimator):
    def infer(self, z):
        start = time.perf_counter()
        u = np.zeros_like(z, dtype=float)
        self.last_trial = u.copy()
        rec = dict(available=False, admitted=False, reason=None)
        if self.sigmas is None:
            rec.update(reason='calibration_unavailable', seconds=time.perf_counter()-start)
            return u, rec, None
        p = stopping.problem(self, z)
        omega = p['omega']
        rec.update(background=p['background'], selected_per_band=omega.sum(axis=1),
                   solver_scale=p['scale'], gradient_scale=p['gradient_scale'])
        if not omega.any():
            rec.update(available=True, reason='empty_multiscale_support')
        else:
            solved = stopping.solve_path(p, qualify=False)
            snapshot = solved['snapshots'].get('declared', solved['final'])
            self.last_trial[self.D] = snapshot['v']*p['scale']
            finite = bool(np.isfinite(self.last_trial).all())
            feasible = bool(finite and np.all(self.last_trial >= 0) and not np.any(self.last_trial[~self.D]))
            ok = solved['operational_stop_available'] and finite and feasible
            rec.update({k:v for k,v in snapshot.items() if k != 'v'})
            rec.update(available=bool(ok), finite=finite, feasible=feasible,
                       reason='accepted_nominal_stop' if ok else 'nominal_stop_unavailable',
                       solver_options=solved['solver_options'], termination=solved['termination'],
                       trace=solved['trace'], solve_wall_seconds=solved['wall_seconds'])
            if ok:
                u[:] = self.last_trial
                rec['admitted'] = bool(np.any(u > 0))
                if not rec['admitted']:
                    rec.update(available=False, reason='exact_zero_solution')
        rec['seconds'] = time.perf_counter()-start
        return u, rec, omega
