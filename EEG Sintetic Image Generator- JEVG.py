import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Esta función genera una figura que representa datos de EEG simulados.
def generate_EEG_figure(channels, duration=10, dt=0.001, noise_level=0.5, dropout_prob=0.001, save_path=None):
    # t es un arreglo de tiempos, desde 0 hasta el tiempo total (duration), incrementando en dt cada paso.
    t = np.arange(0, duration, dt)

    # Crea una figura y los subplots para cada canal.
    fig, axs = plt.subplots(len(channels), figsize=(10, 2*len(channels)))

    # Bucle para recorrer cada canal.
    for i, channel in enumerate(channels):
        # Inicializa el EEG para este canal como un arreglo de ceros.
        EEG = np.zeros_like(t)

        # Añade varias señales sinusoidales de diferentes frecuencias, amplitudes y fases.
        for _ in range(random.randint(5, 15)):
            freq = random.uniform(0.5, 2)
            amp = random.uniform(1, 5)
            phase = random.uniform(0, 2*np.pi)
            EEG += amp * np.sin(2 * np.pi * freq * t + phase)

        # Añade varias señales de tipo spike (pico) de diferentes tiempos, amplitudes y anchos.
        for _ in range(random.randint(10, 20)):
            spike_time = random.uniform(0, duration)
            spike_amp = random.uniform(10, 50)
            spike_width = random.uniform(0.02, 0.07)  
            EEG += spike_amp * np.exp(-0.5 * ((t - spike_time) / spike_width)**2)

        # Añade ruido blanco.
        EEG += np.random.normal(scale=noise_level, size=EEG.shape)

        # Añade interrupciones (dropouts), donde ciertos puntos son simplemente puestos a cero.
        EEG[np.random.random(size=EEG.shape) < dropout_prob] = 0

        # Dibuja el EEG en el subplot correspondiente.
        axs[i].plot(t, EEG)
        axs[i].set_title(f'{channel} Channel', loc='left', pad=-15, y=0.38, backgroundcolor='white')
        axs[i].title.set_size(14)
    
        # Elimina las etiquetas de los ejes x para todos los subplots excepto el último.
        if i < len(channels) - 1:
            axs[i].set_xticklabels([]) 
        else:
            axs[i].set_xlabel('Time (s)')

        # Elimina las etiquetas del eje y para todos los subplots.
        axs[i].set_yticklabels([])

    # Ajusta el layout para que todo se vea bien.
    plt.tight_layout()
    
    # Si se proporcionó una ruta de guardado, guarda la figura como un archivo PNG.
    if save_path is not None:
        plt.savefig(save_path)

    # Muestra la figura.
    plt.show()

# Definir los nombres de los canales
channels = ['C', 'P', 'F', 'T', 'O']

# Crea una carpeta para guardar las imágenes si no existe.
if not os.path.exists('EEG_images'):
    os.makedirs('EEG_images')

# Genera y guarda 5 figuras de EEG.
for i in range(5):
    save_path = f'EEG_images/EEG_{i+1}.png'
    generate_EEG_figure(channels, save_path=save_path)

