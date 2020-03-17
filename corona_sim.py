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
tmax = 2  # Final time
nsteps = 1000
npart = 1000
nvel = 1
D = 2

xx = np.random.random(size=(N, D)) * box
vv = velocity_perturbation(xx, 15)
corona = np.zeros(len(xx)).astype("bool")
death = np.zeros(len(xx)).astype("bool")

for _ in range(10):
    corona[np.random.randint(0, high=len(corona))] = True

# Statements for displaying the results
# --------------------------
fig = plt.figure(figsize=(8, 8))
(ax,) = plt.plot([], [])
plt.axis("on")
time = 0
ims = []

# DATA for plots
infected = []
uninfected = []
dead = []


# Main loop
# ----------------------
for _ in tqdm.tqdm(np.arange(0, nsteps)):

    delta_t = tmax / nsteps
    xx = periodic(xx + delta_t / 2 * vv, box=box)
    time += delta_t

    if np.sum(corona) < N / 2:
        for inf in xx[corona]:
            corona = np.logical_or(corona, (np.linalg.norm(xx - inf, axis=1) < 2e-1))
    else:
        for inf in xx[~corona]:
            corona = np.logical_or(corona, (np.linalg.norm(xx - inf, axis=1) < 2e-1))

    corona = np.logical_and(np.random.random(len(xx)) > 0.10, corona)
    death = np.logical_or(
        np.logical_and(np.random.random(len(xx)) < 0.004, corona), death,
    )
    corona[death] = False
    vv[death] = np.array([0, 0])
    # -------------------------------

    color = np.full(len(xx), "b")
    color[corona] = "r"
    color[death] = "k"
    im = plt.scatter(xx[:, 0], xx[:, 1], animated=True, color=color, marker=".")
    ims.append([im])
    infected.append(np.sum(corona))
    uninfected.append(np.sum(np.logical_and(~corona, ~death)))
    dead.append(np.sum(death))

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ani.save("corona_sim.mp4")

plt.figure()
plt.plot(np.linspace(0, tmax, nsteps), infected, color="r", label="Infected")
plt.plot(np.linspace(0, tmax, nsteps), uninfected, color="b", label="Un-infected")
plt.plot(np.linspace(0, tmax, nsteps), dead, color="k", label="Dead")
plt.xlabel("Time")
plt.ylabel("Number")
plt.legend()
plt.title(f"N:{N} d:{D}")
plt.axhline(y=N, color="y", linestyle="-")
plt.savefig("infected.png", format="png")
plt.show()
