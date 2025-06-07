import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid setup
nx, nz = 200, 200      # Grid size
dx = dz = 10.0         # Grid spacing (m)
nt = 500               # Time steps
dt = 0.001             # Time step (s)

# Physical parameters
v = 2000.0             # Constant velocity (m/s)
Q = 40                 # Quality factor
f0 = 15.0              # Source frequency (Hz)
alpha = np.pi * f0 / Q  # Attenuation coefficient

# CFL condition
cfl = v * dt / dx
assert cfl < 1/np.sqrt(2), f"CFL too high: {cfl:.2f}"

# Wavefields
u = np.zeros((nx, nz))
u_past = np.zeros_like(u)
u_future = np.zeros_like(u)

# Source setup
src_x, src_z = nx // 2, nz // 2
t = np.linspace(-1, 1, int(2/dt))
src_wavelet = (1 - 2*(np.pi*f0*t)**2) * np.exp(-(np.pi*f0*t)**2)

# Storage for animation
frames = []

# Time loop
for it in range(nt):
    # Inject source
    if it < len(src_wavelet):
        u[src_x, src_z] += src_wavelet[it]

    # Finite difference stencil (2nd order)
    lap = (
        -4 * u +
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
    ) / dx**2

    u_future = 2*u - u_past + (v*dt)**2 * lap

    # Apply attenuation
    u_future *= np.exp(-alpha * dt)

    # Shift fields
    u_past, u = u, u_future.copy()

    if it % 5 == 0:
        frames.append(u.copy())

# Animate
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(frames[0], cmap="seismic", vmin=-0.01, vmax=0.01, animated=True)
ax.set_title(f"2D Anelastic Wave Propagation (Q = {Q})")

def update(frame):
    im.set_array(frame)
    return [im]

ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
plt.show()
