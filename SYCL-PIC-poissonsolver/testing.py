import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clean_data(filename):
    """
    Clean and load the particle energy data file.
    Handles potential formatting issues.
    """
    # Read file line by line to handle inconsistencies
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Split by tabs and remove any empty strings
            fields = [x.strip() for x in line.split('\t') if x.strip()]
            if len(fields) >= 6:  # Ensure we have at least the required fields
                try:
                    # Convert to proper types and take only first 6 fields
                    row = [
                        int(fields[0]),          # ID
                        float(fields[1]),        # Energy
                        int(fields[2]),          # Type
                        float(fields[3]),        # Vx
                        float(fields[4]),        # Vy
                        float(fields[5])         # Vz
                    ]
                    data.append(row)
                except (ValueError, IndexError):
                    print(f"Skipping malformed line: {line.strip()}")
    
    # Create DataFrame with cleaned data
    df = pd.DataFrame(data, columns=['ID', 'Energy', 'Type', 'Vx', 'Vy', 'Vz'])
    return df

def plot_histogram(data, bins=100, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """
    Plots a histogram of the given data.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Load and clean the data
df = clean_data('Particle_energy.out')

# Print some statistics
print("\nData Summary:")
print(f"Total particles: {len(df)}")
print(f"Electrons: {len(df[df['Type'] == 0])}")
print(f"Ions: {len(df[df['Type'] == 1])}")
print("\nEnergy Statistics:")
print(df.groupby('Type')['Energy'].describe())

# Plot histograms
df_e = df[df['Type'] == 0]
df_p = df[df['Type'] == 1]

plot_histogram(df_e['Energy'], bins=100, 
              title='Electron Energy Distribution', 
              xlabel='Energy (eV)', 
              ylabel='Count')

plot_histogram(df_p['Energy'], bins=100, 
              title='Ion Energy Distribution', 
              xlabel='Energy (eV)', 
              ylabel='Count')

# Also plot velocity distributions
for component in ['Vx', 'Vy', 'Vz']:
    plot_histogram(abs(df_e[component]), bins=100,
                  title=f'Electron {component} Distribution',
                  xlabel=f'{component} (m/s)',
                  ylabel='Count')