"""
References
[1] "A tutorial on preview control systems", Tabaka (2003).
[2] "hinfsyn: Compute H-infinity optimal controller", Mathworks. 
"""
import scipy
import numpy as np
import control as ct
import scipy.linalg
from . import util


def get_design_system(Ap, Bp, Cp, Dp, Q, R):
    """The design system is constructed from the given
    discrete-time system and cost matrices Q,R. The input
    to the design system is the difference in input to
    the original system; also, d(t) is the difference in
    reference signals. See section 4 in [1]. The output
    of the new system is z = [Q^1/2 e; R^1/2 u].

    The signal dimensions are assumed
        x:              n
        r, d, y, e:     k
        u:              m

    :param Ap: (n,n)
    :param Bp: (n,m)
    :param Cp: (k,n)
    :param Dp: (k,m)
    :param Q: (k,k)
    :param R: (m,m)
    :return:
        A: (n+k,n+k)
        B: (n+k, m)
        C: (k+m, n+k)
        D: (k+m, m)
        E: (n+k, n)
    """
    n = Ap.shape[0]  # no. states
    m = Bp.shape[1]  # no. inputs
    k = Cp.shape[0]  # no. measurements

    A = np.block([
        [Ap, np.zeros(shape=(n, k))],
        [-Cp, np.eye(k)],
    ])
    B = np.block([
        [Bp],
        [-Dp],
    ])
    C = np.block([
        [np.zeros(shape=(k, n)), np.sqrt(Q)],
        [np.zeros(shape=(m, n)), np.zeros(shape=(m, k))],
    ])
    D = np.block([
        [np.zeros(shape=(k, m))],
        [np.sqrt(R)]
    ])
    E = np.block([
        [np.zeros(shape=(n, k))],
        [np.eye(k)]
    ])

    assert A.shape == (n + k, n + k)
    assert B.shape == (n + k, m)
    assert C.shape == (k + m, k + n)
    assert D.shape == (k + m, m)
    assert E.shape == (n + k, k)

    return A, B, C, D, E


def design_system_lqr(A, B, Q, R):
    """Compute lqr feedback for system returned by get_design_system.
    """
    k = Q.shape[0]  # no. errors
    n = A.shape[0] - k  # no. states

    Qd = np.block([
        [np.zeros(shape=(n, n)), np.zeros(shape=(n, k))],
        [np.zeros(shape=(k, n)), Q]
    ])
    assert Qd.shape[0] == A.shape[0]
    assert Qd.shape[1] == A.shape[0]

    K, S, E = ct.dlqr(A, B, Qd, R)  # K is (1, n+k)
    assert K.shape == (1, n+k)

    K_state = K[:, :n]
    K_error = K[:, n:]

    return K_state, K_error


def get_preview_system(A, B, C, D, E, h):
    """Augment the design system with `h` preview vectors. See 3.2 section [1].
    In this function, k is the reference dimension.
    """
    n = A.shape[0]  # state dimension
    m = B.shape[1]  # input dimension
    k = E.shape[1]  # disturbance dimension
    r = k*(h+1)     # length of preview vector
    n_underlying = n - k

    Ad = np.block([
        [np.eye(k) if i==j+1 else np.zeros(shape=(k, k)) for i in range(h+1)] for j in range(h+1)
    ])
    Bd = np.block([
        [np.eye(k)] if i == h else [np.zeros(shape=(k, k))] for i in range(h+1)
    ])

    F = np.block([
        [A, E, np.zeros(shape=(n, k*h))],
        [np.zeros(shape=(r, n)), Ad],
    ])
    G = np.block([
        [B],
        [np.zeros(shape=(r, m))],
    ])
    L = np.block([
        [np.zeros(shape=(n, k))],
        [Bd],
    ])
    H = np.block([C, np.zeros(shape=(k+m, r))])

    assert Ad.shape == (r, r)
    assert Bd.shape == (r, k)
    assert F.shape == (n+r, n+r)
    assert G.shape == (n+r, m)
    assert L.shape == (n+r, k)
    assert H.shape == (k+m, k+n_underlying+r)
    assert D.shape == (k+m, m)

    return F, G, L, H, D


def design_system_lqr_preview(A, B, E, Q, R, h):
    n = A.shape[0]  # state dimension
    m = B.shape[1]  # input dimension
    k = E.shape[1]  # disturbance/error dimension
    r = k*(h+1)     # length of preview vector
    n_underlying = n - k

    Ad = np.block([
        [np.eye(k) if i==j+1 else np.zeros(shape=(k, k)) for i in range(h+1)] for j in range(h+1)
    ])
    F = np.block([
        [A, E, np.zeros(shape=(n, k*h))],
        [np.zeros(shape=(r, n)), Ad],
    ])
    G = np.block([
        [B],
        [np.zeros(shape=(r, m))],
    ])

    Qd = scipy.linalg.block_diag(
        np.zeros(shape=(n_underlying, n_underlying)),
        Q,
        np.zeros(shape=(r, r))
    )
    Rd = R

    assert Qd.shape == (n_underlying+k+r, n_underlying+k+r)
    assert Rd.shape == (m, m)

    K, S, E = ct.dlqr(F, G, Qd, Rd)
    K_x = K[:, :n_underlying]
    K_e = K[:, n_underlying:n_underlying+k]
    K_r = K[:, n_underlying+k:]

    assert K_x.shape == (1, n_underlying)
    assert K_e.shape == (1, k)
    assert K_r.shape == (1, h+1)

    return K_x, K_e, K_r


def preview_system_h_infinity(F, G, L, H, D):
    """todo: hinfsyn not supported yet in python-control
    """
    zeta_dim = F.shape[0]
    ref_dim = L.shape[1]
    u_dim = D.shape[1]

    assert H.shape[0] == D.shape[0]
    assert H.shape[1] == zeta_dim

    P11 = F
    P12 = np.block([L, G])
    P21 = np.block([[H], [np.eye(zeta_dim)]])
    P22 = np.block([
        [np.zeros(shape=(H.shape[0], ref_dim)), D],
        [np.zeros(shape=(zeta_dim, ref_dim)), np.zeros(shape=(zeta_dim, u_dim))]
    ])
    P = ct.ss(P11, P12, P21, P22, dt=True)

    K, CL, gam, rcond = ct.hinfsyn(P, nmeas=zeta_dim, ncon=u_dim)
    K_state = K[:zeta_dim]
    K_e = K[zeta_dim:zeta_dim+ref_dim]
    K_r = K[zeta_dim+ref_dim:]

    assert K_state.shape[1] == zeta_dim
    assert K_r.shape[1] == ref_dim
    assert K_e.shape[1] == ref_dim

    return K_state, K_e, K_r


class Kalman:
    def __init__(self, Ap, Bp, Cp, Dp, Ep, Qn, Rn):
        """Kalman filter wrapper for discrete time system
        x_{t+1} = Ap @ x_t + Bp @ u + Ep @ w_t, y_t = Cp @ x_t + Dp @ u_t + v_t,
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
        self._xhat = self.Ap @ self._xhat + self.Bp @ uk + self._L @ (yk - self.Cp @ self._xhat - self.Dp @ uk)

    def xhat(self):
        return self._xhat


class LQRController:
    def __init__(self, Ap, Bp, Cp, Dp, Q, R):
        ctrb = ct.ctrb(Ap, Bp)
        if ctrb.shape[0] != np.linalg.matrix_rank(ctrb):
            raise ValueError("system not controllable")

        self.e_accum = np.zeros(shape=(Q.shape[0], 1))

        self._A, self._B, self._C, self._D, self._E = get_design_system(Ap, Bp, Cp, Dp, Q, R)
        self.K_state, self.K_error = design_system_lqr(self._A, self._B, Q, R)

    def control(self, x):
        return -self.K_state @ x + -self.K_error @ self.e_accum

    def update(self, error):
        self.e_accum += error

    def K(self):
        return self.K_state
    
    def reset(self):
        self.e_accum = np.zeros_like(self.e_accum)


class LQRPreviewController:
    def __init__(self, Ap, Bp, Cp, Dp, Q, R, h):
        self.h = h
        C = ct.ctrb(Ap, Bp)
        if C.shape[0] != np.linalg.matrix_rank(C):
            raise ValueError("system not controllable")

        self.e_accum = np.zeros(shape=(Q.shape[0], 1))

        A, B, C, D, E = get_design_system(Ap, Bp, Cp, Dp, Q, R)
        self.K_x, self.K_e, self.K_r = design_system_lqr_preview(A, B, E, Q, R, h)
        
    def control(self, x, rs):
        u = self.K_x @ x + self.K_e @ self.e_accum + self.K_r @ rs
        return -u

    def update(self, error):
        self.e_accum += error

    def K(self):
        return self.K_x
    
    def reset(self):
        self.e_accum = np.zeros_like(self.e_accum)


class PreviewController:
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
