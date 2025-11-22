import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------
# 1. Simulation setup
# ------------------------------------------
nx, ny, nz = 60, 60, 60  # grid points
dx = dy = dz = 1.0       # spatial spacing
nt = 500                 # total time steps
dt = 0.0001             # time step for stability; adjust per CFL
v = 3000.0               # wave speed

# CFL check: v*dt/dx < 1/sqrt(3)
cfl = v * dt / dx
cfl_limit = 1/np.sqrt(3)
if cfl >= cfl_limit:
    dt_max = dx / (v * np.sqrt(3))
    raise ValueError(f"CFL condition violated: {cfl:.3f} >= {cfl_limit:.3f}. Use dt < {dt_max:.2e}.")
logger.info(f"Using dt={dt:.2e}, CFL={cfl:.3f} (<{cfl_limit:.3f})")

# Initialize wavefield arrays
u = np.zeros((nx, ny, nz))
u_past = np.zeros_like(u)
u_future = np.zeros_like(u)

# Source: Gaussian wavelet at center
sx, sy, sz = nx//2, ny//2, nz//2
src_wavelet = np.exp(-((np.arange(nt) - 20)/5)**2)

# Storage for midplane slices and full volume snapshots
frames_mid = []
full_snapshots = []
save_full_interval = 10  # store full 3D snapshot every N steps

# Coordinates for plotting
x = np.arange(nx)*dx
y = np.arange(ny)*dy
z = np.arange(nz)*dz
X_phys, Y_phys = np.meshgrid(x, y, indexing='ij')

# ------------------------------------------
# 2. Time-stepping loop
# ------------------------------------------
logger.info("Starting 3D wave simulation...")
for it in range(nt):
    # Compute Laplacian with periodic BC via np.roll
    lap = (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
        np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2) -
        6*u
    ) / dx**2
    # Time update
    u_future = 2*u - u_past + (v*dt)**2 * lap
    # Inject source
    u_future[sx, sy, sz] += src_wavelet[it]
    # Advance fields
    u_past, u = u, u_future
    # Store midplane XY slice
    if it % 2 == 0:
        frames_mid.append(u[:, :, nz//2].copy())
    # Store full snapshot
    if it % save_full_interval == 0:
        full_snapshots.append(u.copy())
logger.info(f"Simulation complete: {len(frames_mid)} midplane frames, {len(full_snapshots)} full snapshots.")

# Amplitude limits
dzlim_mid = max(np.max(np.abs(f)) for f in frames_mid)
dzlim_full = max(np.max(np.abs(snap)) for snap in full_snapshots) if full_snapshots else dzlim_mid

# ------------------------------------------
# 3. Orthogonal slice animation
# ------------------------------------------
if full_snapshots:
    fig_ortho, axes_ortho = plt.subplots(1,3,figsize=(13,4))
    ix, iy, iz = nx//2, ny//2, nz//2
    im_xy = axes_ortho[0].imshow(full_snapshots[0][:,:,iz], cmap='seismic', vmin=-dzlim_full, vmax=dzlim_full, origin='lower')
    axes_ortho[0].set_title('XY slice (Z=mid)')
    im_xz = axes_ortho[1].imshow(full_snapshots[0][:,iy,:].T, cmap='seismic', vmin=-dzlim_full, vmax=dzlim_full, origin='lower')
    axes_ortho[1].set_title('XZ slice (Y=mid)')
    im_yz = axes_ortho[2].imshow(full_snapshots[0][ix,:,:], cmap='seismic', vmin=-dzlim_full, vmax=dzlim_full, origin='lower')
    axes_ortho[2].set_title('YZ slice (X=mid)')
    for ax in axes_ortho:
        ax.axis('off')
    time_text = fig_ortho.suptitle("Time: 0.000 s", fontsize=14)
    def update_ortho(i):
        snap = full_snapshots[i]
        im_xy.set_data(snap[:,:,iz])
        im_xz.set_data(snap[:,iy,:].T)
        im_yz.set_data(snap[ix,:,:])
        t_val = i * save_full_interval * dt
        time_text.set_text(f"Time: {t_val:.3f} s")
        return [im_xy, im_xz, im_yz, time_text]
    ani_ortho = FuncAnimation(fig_ortho, update_ortho,
                          frames=len(full_snapshots),
                          interval=100,
                          blit=False)
    # Shared colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes_ortho[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig_ortho.colorbar(im_xy, cax=cax)
    fig_ortho.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()
    # Optionally save: ani_ortho.save('orthogonal.mp4', writer='ffmpeg', fps=10, dpi=150)

# ------------------------------------------
# 4. PyVista-based visualization
# ------------------------------------------

try:
    import pyvista as pv
    logger.info("PyVista available: setting up isosurface and volume rendering functions.")

    def save_vti(snapshot, filename, dx, dy, dz):
        """Save a 3D snapshot as cell data in VTI format for ParaView."""
        grid = pv.UniformGrid()
        # Dimensions = number of cells + 1 for cell-centered data
        grid.dimensions = np.array(snapshot.shape) + 1
        grid.spacing = (dx, dy, dz)
        grid.origin = (0.0, 0.0, 0.0)
        data_flat = snapshot.flatten(order='F')
        # Assign to cell_data, not point_data
        grid.cell_data['amplitude'] = data_flat
        grid.save(filename)
        logger.info(f"Saved VTI file: {filename} (cell data)")

    def visualize_isosurface(snapshot, iso_fraction=0.5, color='cyan', opacity=0.6):
        """Extract and display isosurface at iso_fraction * max amplitude."""
        zmax = np.max(np.abs(snapshot))
        iso_val = iso_fraction * zmax
        grid = pv.UniformGrid()
        grid.dimensions = np.array(snapshot.shape) + 1
        grid.spacing = (dx, dy, dz)
        grid.origin = (0.0, 0.0, 0.0)
        data_flat = snapshot.flatten(order='F')
        grid.cell_data['amplitude'] = data_flat
        # When contouring, specify the cell data name if needed:
        contours = grid.contour([iso_val], scalars='amplitude')
        p = pv.Plotter()
        p.add_mesh(contours, color=color, opacity=opacity)
        p.add_outline()
        p.add_axes()
        p.show(title=f"Isosurface at {iso_val:.3f}")

    def animate_isosurfaces(snapshots, iso_fraction=0.5, output_folder=None):
        """Animate isosurfaces over time by capturing screenshots and optionally saving."""
        p = pv.Plotter(off_screen=True)
        grid = pv.UniformGrid()
        grid.dimensions = np.array(snapshots[0].shape) + 1
        grid.spacing = (dx, dy, dz)
        grid.origin = (0.0, 0.0, 0.0)
        images = []
        for i, snap in enumerate(snapshots):
            data_flat = snap.flatten(order='F')
            grid.cell_data['amplitude'] = data_flat
            p.clear()
            iso_val = iso_fraction * np.max(np.abs(snap))
            # Specify scalar name if necessary:
            cont = grid.contour([iso_val], scalars='amplitude')
            p.add_mesh(cont, color='cyan', opacity=0.6)
            p.add_outline()
            p.camera_position = 'iso'
            img = p.screenshot()
            images.append(img)
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                fname = os.path.join(output_folder, f"isosurface_{i:03d}.png")
                p.screenshot(fname)
                logger.info(f"Saved screenshot: {fname}")
        try:
            import imageio
            gif_path = 'isosurface_anim.gif'
            imageio.mimsave(gif_path, images, fps=5)
            logger.info(f"Saved GIF animation: {gif_path}")
        except ImportError:
            logger.warning("imageio not available; skipping GIF creation.")
        p.close()

    if full_snapshots:
        save_vti(full_snapshots[-1], 'final_wavefield.vti', dx, dy, dz)
        # visualize_isosurface(full_snapshots[-1])  # interactive, if desired
        animate_isosurfaces(full_snapshots, iso_fraction=0.5, output_folder='isoscreens')

except ImportError:
    logger.warning("PyVista not installed; skipping 3D isosurface visualization.")
