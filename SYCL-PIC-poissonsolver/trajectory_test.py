import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plot_trajectory(filename, particle_id, species_type, i):
    """
    Plot the trajectory of a specific particle over time with domain boundaries.
    
    Parameters:
    - filename: str, path to the trajectory data file
    - particle_id: int, ID of the particle to track
    - species_type: int, type of the particle (0 for electrons, 1 for ions)
    """
    # Load the trajectory data
    df = pd.read_csv(filename, sep='\t', header=None, names=['Time', 'Type', 'ID','Pointer', 'X', 'Y'])
    
    # Filter for the specific particle
    particle_data = df[(df['ID'] == particle_id) & (df['Type'] == species_type)]
    
    if particle_data.empty:
        print(f"No data found for particle ID {particle_id} of type {species_type}.")
        return
    
    # Create figure and plot trajectory
    plt.figure(i, figsize=(10, 6))
    
    # Plot domain boundaries
    plt.axvline(x=0, color='r', linestyle='--', label='Domain boundary')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=1, color='r', linestyle='--')
    
    # Plot particle trajectory
    plt.plot(particle_data['X'], particle_data['Y'], marker='o', label='Particle trajectory')
    
    plt.title(f'Trajectory of Particle ID {particle_id} (Type {species_type})')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

# ...existing code...

filename = 'particle_trajectory.out'
particles = [1,100,4503,5000,480, 9999,60, 6546, 8001, 1080]  # Example particle ID to track
species = [0,1]  # Example species type (0 for electrons, 1 for ions)


for species_type in species:
    for i, particle_id in enumerate(particles):
        plot_trajectory(filename, particle_id, species_type,i)