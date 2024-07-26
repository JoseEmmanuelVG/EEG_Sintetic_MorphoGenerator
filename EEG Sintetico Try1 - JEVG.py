import numpy as np
import matplotlib.pyplot as plt
import random

# Duración total del EEG en segundos
duration = 10

# Tiempo de muestreo (en segundos)
dt = 0.005
t = np.arange(0, duration, dt)

# Generar la señal de EEG
EEG = np.zeros_like(t)

# Generar onda sinusoidal de baja amplitud para los primeros y últimos 1 a 2 segundos
stable_wave_time = random.uniform(1, 2)
stable_wave_indices_start = np.where(t < stable_wave_time)
stable_wave_indices_end = np.where(t > duration - stable_wave_time)
EEG[stable_wave_indices_start] += 0.5 * np.sin(2 * np.pi * 0.5 * t[stable_wave_indices_start])
EEG[stable_wave_indices_end] += 0.5 * np.sin(2 * np.pi * 0.5 * t[stable_wave_indices_end] - np.pi)

# Determinar el tiempo total disponible para las ondas punta-onda lenta
total_time_for_spikes = duration - 2 * stable_wave_time

# Calcular cuántas ondas punta-onda lenta se pueden generar basado en la frecuencia deseada
num_spikes = int(total_time_for_spikes)  # Una onda punta-onda lenta por segundo

# Generar ondas punta-onda lenta
for i in range(num_spikes):
    # Tiempo de inicio para la onda punta-onda lenta
    spike_start_time = stable_wave_time + i * total_time_for_spikes / num_spikes

    # Crear onda punta
    spike_amp = random.uniform(10, 25)
    spike_width = random.uniform(0.05, 0.07)  # Incrementado para ondas más anchas
    spike_indices = np.where((t >= spike_start_time) & (t < spike_start_time + spike_width))
    EEG[spike_indices] += spike_amp * np.exp(-0.5 * ((t[spike_indices] - spike_start_time) / spike_width)**2)

    # Crear onda lenta
    slow_wave_duration = total_time_for_spikes / num_spikes - spike_width
    slow_wave_start_time = spike_start_time + spike_width
    slow_wave_indices = np.where((t >= slow_wave_start_time) & (t < slow_wave_start_time + slow_wave_duration))
    slow_wave_freq = random.uniform(0.5, 2)
    slow_wave_amp = random.uniform(0.8*spike_amp, spike_amp)  # Amplitud cercana a la de la onda de punta
    slow_wave_phase = 0  # Fase de cero para un inicio claro
    EEG[slow_wave_indices] += slow_wave_amp * np.sin(2 * np.pi * slow_wave_freq * (t[slow_wave_indices] - slow_wave_start_time) + slow_wave_phase)

# Graficar la señal de EEG
plt.figure(figsize=(10, 4))
plt.plot(t, EEG)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Simulated EEG')
plt.show()
