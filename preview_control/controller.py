"""
References
[1] "A tutorial on preview control systems", Tabaka (2003).
[2] "hinfsyn: Compute H-infinity optimal controller", Mathworks. 
"""

import numpy as np
import control as ct


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
        [np.eye(n=k)]
    ])

    shapes = {
        A: (n + k, n + k),
        B: (n + k, m),
        C: (n + k, k + m),
        D: (k + m, m),
        E: (n + k, k),
    }

    for matrix, shape in shapes.items():
        assert matrix.shape == shape, f"has shape {matrix.shape}, should have shape {shape}"

    return A, B, C, D, E


def design_system_lqr(A, B, Q, R):
    """Compute lqr feedback for system returned by
    get_design_system.
    """
    n = A.shape[0]  # no. states
    m = B.shape[1]  # no. inputs
    k = Q.shape[0]  # no. errors

    Qd = np.block([
        [np.zeros(shape=(n, n)), np.zeros(shape=(n, k))],
        [np.zeros(shape=(k, n)), Q]
    ])
    assert Qd.shape[0] == n+k
    assert Qd.shape[1] == n+k

    K, S, E = ct.dlqr(A, B, Qd, R)  # K is (1, n+k)
    assert K.shape == (1, n+k)

    K_state = K[:, :n]
    K_error = K[:, n:]

    return K_state, K_error


def get_preview_system(A, B, C, D, E, h):
    """Augment the provided system with the preview vectors.
    See 3.2 section [1].

    :param A: (n,n)
    :param B: (n,m)
    :param C: (k,n)
    :param D: (k,m)
    :param E: (k,1)
    :param h: number of preview steps
    :return: _description_
    """
    n = A.shape[0]  # no. states
    m = B.shape[1]  # no. inputs
    k = C.shape[0]  # no. measurements
    r = k*(h+1)     # length of preview vector

    Ad = np.block([
        [np.eye(k) if i==j+1 else np.zeros(shape=(k, k)) for i in range(h+1)] for j in range(h+1)
    ])
    Bd = np.block([
        [np.eye(k)] if i == h else np.zeros(shape=(k, k)) for i in range(h+1)
    ])

    F = np.block([
        [A, E, np.zeros(shape=(n, k*h))],
        [np.zeros(shape=(r, r)), Ad],
    ])
    G = np.block([
        [B],
        [np.zeros(shape=(r, m))],
    ])
    L = np.block([
        [np.zeros(shape=(n, k))],
        [Bd],
    ])
    H = np.block([
        [C],
        [np.zeros(shape=(k, r))]
    ])

    shapes = {
        F: (n + r, n + r),
        G: (n + r, m),
        L: (n + r, k),
        H: (k, n + r),
        D: (k, m),
        Ad: (r, r),
        Bd: (r, k),
    }

    for matrix, shape in shapes.items():
        assert matrix.shape == shape, f"has shape {matrix.shape}, should have shape {shape}"

    return F, G, L, H, D


def preview_system_h_infinity(F, G, L, H, D, zeta_dim, ref_dim, u_dim):
    assert H.shape[0] == D.shape[0]

    P11 = F
    P12 = np.block([L, G])
    P21 = np.block([[H, np.eye(zeta_dim)]])
    P22 = np.block([
        [np.zeros(shape=(H.shape[0], ref_dim)), D],
        [np.zeros(shape=(zeta_dim, ref_dim)), np.zeros(shape=(zeta_dim, u_dim))]
    ])
    P = ct.ss(P11, P12, P21, P22)

    K, CL, gam, rcond = ct.hinfsyn(P, nmeas=zeta_dim, ncon=u_dim)
    K_state = K[:zeta_dim]
    K_e = K[zeta_dim:2*zeta_dim]
    K_r = K[2*zeta_dim:]

    return K_state, K_e, K_r


class LQRController:
    def __init__(self, Ap, Bp, Cp, Dp, Q, R):
        self.e_accum = 0

        A, B, C, D, E = get_design_system(Ap, Bp, Cp, Dp, Q, R)
        self.K_state, self.K_error = design_system_lqr(A, B, Q, R)
        
    def control(self, x):
        return self.K_state @ x + self.K_error @ self.e_accum

    def update(self, error):
        self.e_accum += error


class PreviewController:
    def __init__(self, Ap, Bp, Cp, Dp, Q, R):
        self.e_accum = 0

        A, B, C, D, E = get_design_system(Ap, Bp, Cp, Dp, Q, R)

    def control(self, x):
        pass

    def update(self, error):
        self.e_accum += error
