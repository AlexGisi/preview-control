import numpy as np
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


def choose_observer_poles(plant_poles, factor=0.5):
    """
    Given an array of plant poles (complex eigenvalues),
    return a new array of observer poles with smaller magnitudes
    to ensure faster estimation dynamics.
    
    Parameters
    ----------
    plant_poles : array-like of complex
        Closed-loop plant poles, e.g. np.linalg.eig(Ap - Bp @ K)[0].
    factor : float
        Factor by which to reduce the magnitude of each pole. 
        (Example: 0.5 => each observer pole is half the radius 
         of the corresponding plant pole, same angle.)

    Returns
    -------
    obs_poles : np.ndarray
        A new set of poles to use for the Luenberger observer, 
        all strictly inside the unit circle.
    """
    obs_poles = []
    for p in plant_poles:
        mag = np.abs(p)
        ang = np.angle(p)
        
        new_mag = factor * mag
        
        if new_mag >= 1.0:
            new_mag = 0.9
        
        obs_poles.append(new_mag * np.exp(1j * ang))
    
    return np.array(obs_poles)
