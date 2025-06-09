# 🌊 3D Anelastic Wave Simulation (WebGL)

This project visualizes a simplified **anelastic seismic wave propagation** using `three.js`, incorporating **attenuation (Q)** and **frequency-controlled source** in real time.

![screenshot](preview.png)

## 🚀 Features

- 🔷 3D wave propagation visualized with cube heights and colors.
- 🎛️ Real-time controls using `dat.GUI`:
  - **Q factor** (attenuation)
  - **Source frequency**
  - **Radial decay**
- 🌐 Fully browser-based (no backend or server required).
- ⚡ Runs smoothly in modern browsers via WebGL.

## 🎮 Demo Controls

- **Q**: Lower Q → faster attenuation
- **Frequency**: Controls wave source frequency (Hz)
- **Decay Factor**: Controls radial amplitude decay

## 🧪 Physics Simplification

This is a visual approximation:
- Uses exponential decay to mimic constant-Q attenuation:
  

- No true PDE solving (for performance).

## 📦 Usage

1. Clone or download the repo.
2. Open `anelastic_3d_gui_simulation.html` in any browser.
3. Adjust parameters via the GUI to explore wave behavior.

## 🔧 Tech Stack

- `three.js` – 3D rendering
- `dat.GUI` – Control panel
- HTML5 + JavaScript

## 📁 Files

- `anelastic_3d_gui_simulation.html` – Main simulation file
- `preview.png` – (Optional) Screenshot for display in README

## 📜 License

MIT License. Attribution appreciated but not required.
