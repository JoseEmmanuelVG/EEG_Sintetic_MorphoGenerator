import numpy as np
import matplotlib.pyplot as plt
import random

# Duración total del EEG en segundos
duration = 10

# Tiempo de muestreo (en segundos)
dt = 0.001
t = np.arange(0, duration, dt)

# Definir los nombres de los canales
channels = ['C', 'P', 'F', 'T', 'O']  # Central, Parietal, Frontal, Temporal, Occipital

# Crear una figura para los gráficos
fig, axs = plt.subplots(len(channels), figsize=(10, 2*len(channels)))

for i, channel in enumerate(channels):
    # Generar la señal de EEG como una suma de ondas lentas y rápidas
    EEG = np.zeros_like(t)

    # Agregar ondas lentas (0.5 - 2 Hz, según literatura de EEG)
    for _ in range(random.randint(5, 15)):
        freq = random.uniform(0.5, 2)
        amp = random.uniform(1, 5)
        phase = random.uniform(0, 2*np.pi)
        EEG += amp * np.sin(2 * np.pi * freq * t + phase)

    # Agregar ondas punta en ubicaciones aleatorias
    for _ in range(random.randint(10, 20)):
        spike_time = random.uniform(0, duration)
        spike_amp = random.uniform(10, 50)
        spike_width = random.uniform(0.02, 0.07)  # Ajustar la duración del spike a 20-70 ms
        EEG += spike_amp * np.exp(-0.5 * ((t - spike_time) / spike_width)**2)

    # Graficar la señal de EEG
    axs[i].plot(t, EEG)
    axs[i].set_title(f'{channel} Channel', loc='left', pad=-15, y=0.38, backgroundcolor='white')
    axs[i].title.set_size(14)
    
    # Eliminar etiquetas del eje x excepto para la última gráfica
    if i < len(channels) - 1:
        axs[i].set_xticklabels([]) 
    else:
        axs[i].set_xlabel('Time (s)')

    # Eliminar etiquetas del eje y
    axs[i].set_yticklabels([])

plt.tight_layout()
plt.show()
