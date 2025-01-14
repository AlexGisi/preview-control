"""
Based off example 4 in "A Tutorial on Preview Control Systems", Takaba (2003).
"""
import controller
import numpy as np
import matplotlib.pyplot as plt


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
Cp = np.array([[0.0, 1.0, 0.0, 0.0]])
Dp = np.array([[0.0]])

# Use a step reference
times = range(80)
rs = [np.zeros(shape=(Cp.shape[0], 1)) if t < 5 else np.ones(shape=(Cp.shape[0], 1)) for t in times]

# Synthesize lqr controller
Q = np.array([[20.0]])
R = np.array([[1.0]])
lqr = controller.LQRController(Ap, Bp, Cp, Dp, Q, R)

# Simulate response
x = np.zeros(shape=(Ap.shape[0], 1))
xs = [x]
ys = [Cp @ x]
for t in times[:-1]:
    u = lqr.control(x)
    x = Ap @ x + Bp @ u
    y = Cp @ x + Dp @ u
    e = rs[t] - y
    lqr.update(e)
    xs.append(x)
    ys.append(y)

# Assume scalar output, reference
ys = [float(y.item()) for y in ys]
rs = [float(r.item()) for r in rs]

# Plot result
plt.plot(times, ys, label='y')
plt.plot(times, rs, 'k--', label='r')
plt.legend()
plt.show()
