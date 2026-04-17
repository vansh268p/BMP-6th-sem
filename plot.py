import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# === Directory containing ion density data ===
base_dir = "out/Density/"
file_pattern = os.path.join(base_dir, "*")
files = sorted(glob.glob(file_pattern), key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))

# === Load the last file ===
last_file = files[36]
print(f"Loading: {last_file}")
data = np.loadtxt(last_file)

# === Grid settings ===
Ny, Nx = data.shape  # Ny = rows (y), Nx = columns (x)

Lx = 0.025   # domain length in x [m]
Ly = 0.0128   # domain length in y [m]

dx = Lx / Nx
dy = Ly / Ny
DT_SEC = 5e-12

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# === Plot the 2D ion density profile ===
plt.figure(figsize=(8, 3))
plt.imshow(
    data,
    origin='lower',
    cmap='rainbow',
    aspect='auto',
    extent=[0, Lx, 0, Ly],   # [xmin, xmax, ymin, ymax] in meters
    #vmax=1e18
)
raw_iter = int(os.path.basename(last_file))
t_us = raw_iter * DT_SEC * 1e6
plt.title(f"t = {t_us:.4f} µs")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.colorbar(label="Plasma Density [m$^{-3}$]")
plt.tight_layout()
plt.savefig("elec_density.png")
