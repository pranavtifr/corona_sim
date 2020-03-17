#! /usr/bin/env python
"""Corona virus simulation."""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tqdm


def periodic(xx, box=10):
    """Enforce wraps xx to enforce periodic boundary conditions."""
    xx[xx < 0] += box
    xx[xx >= box] -= box
    return xx


def velocity_perturbation(xx, maxvel=5):
    """Assign inital perturbation in velocity."""
    vv = np.random.randn(*xx.shape) * maxvel
    return vv


# Let's set the parameters
# ------------------------------
N = 1000
box = 10.0  # Size of our box
tmax = 5.0  # Final time
nsteps = 3000
npart = 1000
nvel = 1
D = 2

xx = np.random.random(size=(N, D)) * box
vv = velocity_perturbation(xx, 5)
corona = np.zeros(len(xx)).astype("bool")

for _ in range(10):
    corona[np.random.randint(0, high=len(corona))] = True

# Statements for displaying the results
# --------------------------
fig = plt.figure(figsize=(8, 8))
(ax,) = plt.plot([], [])
plt.axis("on")
time = 0
ims = []
color = np.full(len(xx), "b")

# DATA for plots
infected = []


# Main loop
# ----------------------
for _ in tqdm.tqdm(np.arange(0, nsteps)):

    delta_t = tmax / nsteps
    xx = periodic(xx + delta_t / 2 * vv, box=box)
    time += delta_t

    for inf in xx[corona]:
        corona = np.logical_or(corona, (np.linalg.norm(xx - inf, axis=1) < 1e-1))

    # -------------------------------
    color[corona] = "r"
    # im = plt.scatter(xx[:, 0], xx[:, 1], animated=True, color=color)
    # ims.append([im])
    infected.append(np.sum(corona))

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# ani.save("phase_space_1.mp4")
plt.plot(infected)
plt.xlabel("time")
plt.ylabel("infected")
plt.savefig("infected.png", format="png")
plt.show()
