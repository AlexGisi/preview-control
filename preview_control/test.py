"""
Based off example 4 in "A Tutorial on Preview Control Systems", Takaba (2003).
"""
import controller
import numpy as np

DT = 0.05

# Define the system
Ap = np.array([
    [0.9752, 0.0248, 0.1983, 0.0017],
    [0.0248, 0.9752, 0.0017, 0.1983],
    [-0.2459, 0.2459, 0.9752, 0.0248],
    [0.2459, -0.2459, 0.0248, 0.9752],
])
Bp = np.array([
    [-0.0199],
    [-0.0001],
    [-0.1983],
    [-0.0017],
])
Cp = np.array([0.0, 1.0, 0.0, 0.0])
Dp = 0.0

# Synthesize lqr controller
Q = 20
R = 1
A, B, C, D, E = controller.get_design_system(Ap, Bp, Cp, Dp, Q, R)

# Simulate response

# Plot result

