from scipy.integrate import quad_vec
from scipy.linalg import expm


def discretized_lti(A, B, dt, max_error=0.01):
    """Return the matrices Ad, Bd which solve
    the discrete time system 
    x_{k+1} = Ad @ x_k + Bd @ u_k 
    for the continuous time LTI system defined by A, B
    """
    Ad = expm(A * dt)

    integrand = lambda tau: expm(A * (dt - tau)) @ B
    Bd, error = quad_vec(integrand, 0, dt)

    if error > max_error:
        raise ValueError()
    
    return Ad, Bd
