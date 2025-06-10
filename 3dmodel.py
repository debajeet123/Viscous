import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm
import os
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

plt.switch_backend('tkagg')  # GUI-compatible backend

# ----------------------------
# 1. GRID SETUP & SIMULATION PARAMETERS
# ----------------------------
nx, nz = 200, 200
dx = dz = 10.0
nt = 5000
dt = 0.001

# Physical parameters
v = 4000.0
Q = 40
f0 = 15.0
alpha = np.pi * f0 / Q

# CFL condition
cfl = v * dt / dx
assert cfl < 1/np.sqrt(2), f"CFL too high: {cfl:.2f}"

# Precompute meshgrid for plotting after simulation:
x_idx = np.arange(nx)
z_idx = np.arange(nz)
X_idx, Z_idx = np.meshgrid(x_idx, z_idx, indexing='ij')  # shape (nx, nz)
X_phys = X_idx * dx
Y_phys = Z_idx * dz  # we treat the second spatial index as “Y” for plotting

# We'll fill `frames` during time-stepping, then compute zlim:
frames = []

# ----------------------------
# 2. STYLE FUNCTION (takes grid & zlim as args)
# ----------------------------
def style_ax(ax, X_phys, Y_phys, zlim):
    """
    Apply black-background + white-wireframe styling to a 3D Axes `ax`.
    X_phys, Y_phys are used only to set x/y limits; zlim for z limits.
    """
    # Black background
    ax.set_facecolor('black')
    # Hide panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Tick label colors
    ax.w_xaxis.set_tick_params(color='white', labelcolor='white')
    ax.w_yaxis.set_tick_params(color='white', labelcolor='white')
    ax.w_zaxis.set_tick_params(color='white', labelcolor='white')
    # Axis label colors
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    # Turn off default gridlines (we rely on wireframe plotting)
    ax.grid(False)
    # Set limits from the grid extents and amplitude
    ax.set_xlim(X_phys.min(), X_phys.max())
    ax.set_ylim(Y_phys.min(), Y_phys.max())
    ax.set_zlim(-zlim, zlim)

# ----------------------------
# 3. WIREFRAME PLOTTING FUNCTIONS
# ----------------------------
def plot_single_wireframe(frame_array, X_phys, Y_phys, zlim, elev=30, azim=135, stride=5):
    """
    Plot a single wireframe snapshot of frame_array (shape nx x nz).
    - X_phys, Y_phys: the meshgrid arrays (same shape as frame_array).
    - zlim: max amplitude for symmetric z limits.
    - elev, azim: view angles.
    - stride: how many grid points to skip for wireframe density (larger => sparser).
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    style_ax(ax, X_phys, Y_phys, zlim)

    # Labels or minimal look
    ax.set_xlabel("X", color='white')
    ax.set_ylabel("Y", color='white')
    ax.set_zlabel("Amplitude", color='white')
    ax.set_title("Wavefield Wireframe Snapshot", color='white')

    # Plot wireframe: note correct order X_phys, Y_phys, frame_array
    ax.plot_wireframe(
        X_phys, Y_phys, frame_array,
        rstride=stride, cstride=stride,
        color='white', linewidth=0.5
    )
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()

def animate_wireframe(frames, X_phys, Y_phys, zlim,
                      elev=30, azim=135, stride=5, interval=50, output_path=None):
    """
    Create (and optionally save) an animation of wireframe evolving through `frames`.
    - frames: list or array of 2D arrays shape (nx, nz).
    - X_phys, Y_phys: meshgrid arrays.
    - zlim: max amplitude for symmetric z limits.
    - elev, azim: initial view angles.
    - stride: wireframe density.
    - interval: ms between frames.
    - output_path: if given (string), save animation to this path via ffmpeg.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')

    # Initial styling and initial plot:
    style_ax(ax, X_phys, Y_phys, zlim)
    ax.set_xlabel("X", color='white')
    ax.set_ylabel("Y", color='white')
    ax.set_zlabel("Amplitude", color='white')

    # Plot the first frame
    Z0 = frames[0]
    ax.plot_wireframe(
        X_phys, Y_phys, Z0,
        rstride=stride, cstride=stride,
        color='white', linewidth=0.5
    )
    ax.set_title("Frame 0", color='white')
    ax.view_init(elev=elev, azim=azim)

    def update(frame_idx):
        ax.clear()
        # Reapply styling after clear:
        fig.patch.set_facecolor('black')
        style_ax(ax, X_phys, Y_phys, zlim)
        ax.set_xlabel("X", color='white')
        ax.set_ylabel("Y", color='white')
        ax.set_zlabel("Amplitude", color='white')
        ax.set_title(f"Frame {frame_idx}", color='white')
        ax.view_init(elev=elev, azim=azim)
        Z = frames[frame_idx]
        ax.plot_wireframe(
            X_phys, Y_phys, Z,
            rstride=stride, cstride=stride,
            color='white', linewidth=0.5
        )
        return

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)

    if output_path:
        # Save via ffmpeg. Make sure ffmpeg is installed in your environment.
        fps = max(1, int(1000 / interval))
        ani.save(output_path, writer='ffmpeg', fps=fps, dpi=150)
        print(f"Animation saved to {output_path}")
    else:
        plt.show()

    return ani

# ----------------------------
# 4. SIMULATION: time-stepping and collect frames
# ----------------------------
# Prepare wavefields
u = np.zeros((nx, nz))
u_past = np.zeros_like(u)
u_future = np.zeros_like(u)

# Source setup
src_x, src_z = nx // 2, nz // 2
t_wave = np.linspace(-1, 1, int(2/dt))
src_wavelet = (1 - 2*(np.pi*f0*t_wave)**2) * np.exp(-(np.pi*f0*t_wave)**2)
src_wavelet /= np.max(np.abs(src_wavelet))
if len(src_wavelet) < nt:
    src_wavelet = np.pad(src_wavelet, (0, nt - len(src_wavelet)), 'constant')

# Optional: plot the wavelet
plt.figure(figsize=(8, 4))
plt.plot(np.linspace(-1, 1, len(src_wavelet)), src_wavelet, color='darkblue')
plt.title("Source Wavelet", fontsize=14)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Time stepping: store every 5th step (so total frames ~ nt/5)
for it in tqdm.tqdm(range(nt)):
    lap = (
        -4 * u +
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
    ) / dx**2

    u_future = 2*u - u_past + (v*dt)**2 * lap
    u_future *= np.exp(-alpha * dt)

    if it < len(src_wavelet):
        u_future[src_x, src_z] += src_wavelet[it]

    u_past = u
    u = u_future

    if it % 5 == 0:
        frames.append(u.copy())

# After collecting frames, compute amplitude limit:
amp_max = max(np.max(np.abs(f)) for f in frames)
zlim = amp_max  # symmetric limits [-zlim, +zlim]

# ----------------------------
# 5. PLOTTING: 3D SURFACE (optional) and WIREFRAME
# ----------------------------
# Example: 3D surface of final snapshot
frame_to_plot = frames[-1]
fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
style_ax(ax3d, X_phys, Y_phys, zlim)

# Surface plot: ensure correct order X_phys, Y_phys, Z
surf = ax3d.plot_surface(
    X_phys, Y_phys, frame_to_plot,
    cmap='seismic',
    linewidth=0,
    antialiased=True
)
m = plt.cm.ScalarMappable(cmap='seismic')
m.set_array(frame_to_plot)
cbar = fig.colorbar(m, shrink=0.5, aspect=10)
cbar.set_label("Amplitude")
ax3d.set_xlabel("X (m)", color='white')
ax3d.set_ylabel("Y (m)", color='white')
ax3d.set_zlabel("Amplitude", color='white')
ax3d.set_title("3D Surface Plot of Wavefield (Final Snapshot)", color='white')
ax3d.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()

# Example: single wireframe snapshot
plot_single_wireframe(frames[-1], X_phys, Y_phys, zlim, elev=30, azim=135, stride=5)

# Example: animate wireframe over time and save
# Note: this can be slow for many frames; you may choose fewer frames or larger stride.
ani = animate_wireframe(
    frames,
    X_phys, Y_phys, zlim,
    elev=30, azim=135,
    stride=5,       # adjust density
    interval=50,    # ms between frames
    output_path="wave_wireframe.mp4"
)

# Example: 2D imshow animation (as before)
vlim = np.max(np.abs(frames[-1])) * 0.5
fig2, ax2 = plt.subplots(figsize=(7, 6))
im = ax2.imshow(frames[0], cmap="seismic", vmin=-vlim, vmax=vlim, animated=True)
time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, color='white', fontsize=12,
                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

def update2(frame_index):
    frame = frames[frame_index]
    im.set_array(frame)
    time_text.set_text(f'Time: {frame_index * 5 * dt:.2f} s')
    return [im, time_text]

ani2 = FuncAnimation(fig2, update2, frames=len(frames), interval=30, blit=True)
ax2.set_title(f"2D Anelastic Wave Propagation (Q = {Q}, f₀ = {f0} Hz)", color='white')
plt.tight_layout()
ani2.save("anelastic_wave_propagation.mp4", writer='ffmpeg', fps=30, dpi=200)
plt.show()
plt.close(fig2)

if os.path.exists("anelastic_wave_propagation.mp4"):
    print("✅ 2D animation saved successfully.")
else:
    print("❌ Failed to save the 2D animation.")
