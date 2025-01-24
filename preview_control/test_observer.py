"""
Based off example 4 in "A Tutorial on Preview Control Systems", Takaba (2003).
"""
from . import controller
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
ts = range(80)
rs = [np.zeros(shape=(Cp.shape[0], 1)) if t < 25 else np.ones(shape=(Cp.shape[0], 1)) for t in ts]
get_ref = lambda i, h: np.array([rs[j] if j<len(rs) else rs[-1] for j in range(i, i+h+1)]).reshape(-1, 1)

# Define control objective function
Q = np.array([[20.0]])
R = np.array([[1.0]])


def simulate(controller, times, get_ref):
    n_preview = controller.h if hasattr(controller, 'h') else None 

    # x = np.zeros(shape=(Ap.shape[0], 1))
    x = np.ones(shape=(Ap.shape[0], 1))
    xs = [x]
    ys = [Cp @ x]
    us = []
    for t in times[:-1]:
        u = controller.control(x, rs=get_ref(t, n_preview)) if n_preview else controller.control(x)
        x = Ap @ x + Bp @ u
        y = Cp @ x + Dp @ u
        e = rs[t] - y
        controller.update(e)
        xs.append(x)
        ys.append(y)
        us.append(u)

    # Assume scalar inputs, measurements
    ys = [float(y.item()) for y in ys]
    us = [float(u.item()) for u in us]

    return xs, ys, us


def simulate_observer(controller, times, get_ref, observer):
    n_preview = controller.h if hasattr(controller, 'h') else None 

    x = np.ones(shape=(Ap.shape[0], 1))
    xs = [x]
    xos = [observer.xhat()]
    ys = [Cp @ x]
    us = []
    for t in times[:-1]:
        # Estimate the state
        xhat = observer.xhat()

        # Apply control
        u = controller.control(xhat, rs=get_ref(t, n_preview)) if n_preview else controller.control(xhat)
        x = Ap @ x + Bp @ u
        y = Cp @ x + Dp @ u

        # Update observer
        observer.update(y, u)

        # Bookkeeping
        e = rs[t] - y
        controller.update(e)
        xs.append(x)
        xos.append(observer.xhat())
        ys.append(y)
        us.append(u)

    # Assume scalar inputs, measurements
    ys = [float(y.item()) for y in ys]
    us = [float(u.item()) for u in us]

    return xs, xos, ys, us

# ctrl = controller.LQRPreviewController(Ap, Bp, Cp, Dp, Q, R, h=1)
ctrl = controller.LQRController(Ap, Bp, Cp, Dp, Q, R)
# ctrl = controller.SimpleController(Ap, Bp, Cp, Dp)
obsrv = controller.Observer(ctrl._A, ctrl._B, ctrl._C, ctrl._D, ctrl.K())
# obsrv = controller.Observer(Ap, Bp, Cp, Dp, ctrl.K())

xs, xos, ys, us = simulate_observer(ctrl, ts, get_ref, obsrv)
ctrl.reset()
xs_nob, ys_nob, us_nob = simulate(ctrl, ts, get_ref)

rs = [float(r.item()) for r in rs]


fig, axs = plt.subplots(3, 1)

axs[0].plot(ts, ys, label=f'y_observer')
axs[0].plot(ts, ys_nob, '--', label=f'y')
axs[0].plot(ts, rs, 'k--', label='r')
axs[0].legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))  # Move legend to the far left
axs[0].set_title('output')

axs[1].plot(ts[:-1], us, label=f'u_observer')
axs[1].plot(ts[:-1], us_nob, '--', label=f'u')
axs[1].legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))  # Move legend to the far left
axs[1].set_title('control input')

axs[2].plot(ts, [np.linalg.norm(a - o) for a, o in zip(xs, xos)], label='||xhat-x||')
axs[2].legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))
axs[2].set_title('estimation error')

plt.tight_layout()
plt.show()
