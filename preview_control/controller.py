"""
References
[1] "hinfsyn: Compute H-infinity optimal controller", Mathworks. 
"""
import scipy
import numpy as np
import control as ct
import scipy.linalg
from . import system_construction


class Kalman:
    def __init__(self, Ap, Bp, Cp, Dp, Ep, Qn, Rn):
        """Kalman filter wrapper for discrete time system
                x_{t+1} = Ap @ x_t + Bp @ u + Ep * w_t,
                y_t = Cp @ x_t + Dp @ u_t + v_t,
        where w_t is noise with covariance matrix Qn and v_t is noise with covariance
        matrix Rn.

        :param Ap: _description_
        :param Bp: _description_
        :param Cp: _description_
        :param Dp: _description_
        :param Ep: _description_
        :param Qn: _description_
        :param Rn: _description_
        """
        self.Ap = Ap
        self.Bp = Bp
        self.Cp = Cp
        self.Dp = Dp
        self.Ep = Ep
        self._xhat = np.zeros(shape=(Ap.shape[0], 1))

        self._L, _, _ = ct.dlqe(Ap, Ep, Cp, Qn, Rn)

        assert self._L.shape == self._xhat.shape

    def update(self, yk, uk):
        if not isinstance(yk, np.ndarray) and isinstance(yk, float):
            yk = np.array([[yk]])
        if not isinstance(uk, np.ndarray) and isinstance(uk, float):
            uk = np.array([[uk]])


        self._xhat = self.Ap @ self._xhat + self.Bp @ uk + self._L @ (yk - self.Cp @ self._xhat - self.Dp @ uk)

    def xhat(self):
        return self._xhat
    
    def yhat(self):
        return self.Cp @ self._xhat
    
    
class SimpleController:
    def __init__(self, Ap, Bp, Q, R):
        ctrb = ct.ctrb(Ap, Bp)
        if ctrb.shape[0] != np.linalg.matrix_rank(ctrb):
            raise ValueError("system not controllable")

        self.K_state, S, E = ct.dlqr(Ap, Bp, Q, R)
        
    def control(self, x):
        return -self.K_state @ x

    def update(self, error):
        pass

    def K(self):
        return self.K_state
    
    def reset(self):
        pass
    


class LQIController:
    def __init__(self, Ap, Bp, Cp, Dp, Q, R):
        ctrb = ct.ctrb(Ap, Bp)
        if ctrb.shape[0] != np.linalg.matrix_rank(ctrb):
            raise ValueError("system not controllable")

        self.e_accum = np.zeros(shape=(Q.shape[0], 1))

        self._A, self._B, self._C, self._D, self._E = system_construction.get_design_system(Ap, Bp, Cp, Dp, Q, R)
        self.K_state, self.K_error = system_construction.design_system_lqi(self._A, self._B, Q, R)

    def control(self, x):
        return -self.K_state @ x + -self.K_error @ self.e_accum

    def update(self, error):
        self.e_accum += error

    def K(self):
        return self.K_state
    
    def reset(self):
        self.e_accum = np.zeros_like(self.e_accum)


class LQIPreviewController:
    def __init__(self, Ap, Bp, Cp, Dp, Q, R, h):
        self.h = h
        C = ct.ctrb(Ap, Bp)
        if C.shape[0] != np.linalg.matrix_rank(C):
            raise ValueError("system not controllable")

        self.e_accum = np.zeros(shape=(Q.shape[0], 1))

        A, B, C, D, E = system_construction.get_design_system(Ap, Bp, Cp, Dp, Q, R)
        self.K_x, self.K_e, self.K_r = system_construction.design_system_lqi_preview(A, B, E, Q, R, h)
        
    def control(self, x, rs):
        u = self.K_x @ x + self.K_e @ self.e_accum + self.K_r @ rs
        return -u

    def update(self, error):
        self.e_accum += error

    def K(self):
        return self.K_x
    
    def reset(self):
        self.e_accum = np.zeros_like(self.e_accum)


class HInfPreviewController:
    def __init__(self, Ap, Bp, Cp, Dp, Q, R, h):
        raise NotImplementedError("must implement hinfsys in python-control")

        C = ct.ctrb(Ap, Bp)
        if C.shape[0] != np.linalg.matrix_rank(C):
            raise ValueError("system not controllable")
        
        self.h = h
        self.e_accum = 0

        sys = get_design_system(Ap, Bp, Cp, Dp, Q, R)
        preview_sys = get_preview_system(*sys, h)
        self.K_state, self.K_e, self.K_r = preview_system_h_infinity(*preview_sys)

    def control(self, x, rs):
        u = self.K_state @ x + self.K_e @ self.e_accum + sum([ki @ ri for ki, ri in zip(self.K_r, rs)])
        return -u

    def update(self, error):
        self.e_accum += error
