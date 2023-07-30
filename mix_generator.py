import numpy as np
import random
from scipy.signal import gaussian
import matplotlib.pyplot as plt


def generate_spike(amplitude_spike, duration, sfreq):
    n_samples = int(duration * sfreq)
    spike = gaussian(n_samples, std=n_samples/7)
    return amplitude_spike * spike / np.max(spike)

def generate_slow_wave(amplitude_wave, duration, sfreq):
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq
    slow_wave = np.sin(1.9 * np.pi * 1.9 * t)  # slow wave frequency set to 2 Hz (Delta wave)
    return amplitude_wave * slow_wave

def generate_spike_wave_group(sfreq, group_duration = 3): # group_duration in seconds
    n_samples = int(group_duration * sfreq)
    group_data = np.zeros(n_samples)
    n_spikes = random.randint(9, 12)  # 8-12 spikes
    current_start_index = 0  # Start index for the first spike-wave pair
    for _ in range(n_spikes):
        amplitude_spike = random.uniform(0.5, 1.0) * 100
        duration_spike = random.uniform(0.02, 0.07)
        n_duration_spike = int(duration_spike * sfreq)
        spike = generate_spike(amplitude_spike, duration_spike, sfreq)

        if random.random() < 0.8:
            amplitude_wave = random.uniform(0.5, amplitude_spike / 100) * 100
        else:
            amplitude_wave = random.uniform(0.5, 1.1) * 100

        duration_wave = random.uniform(0.22, 0.33)
        n_duration_wave = int(duration_wave * sfreq)
        slow_wave = generate_slow_wave(amplitude_wave, duration_wave, sfreq)

        if current_start_index + n_duration_spike + n_duration_wave <= len(group_data):
            group_data[current_start_index:current_start_index + n_duration_spike] += spike
            group_data[current_start_index + n_duration_spike:current_start_index + n_duration_spike + n_duration_wave] += slow_wave
            current_start_index += n_duration_spike + n_duration_wave  # Update start index for the next spike-wave pair
    return group_data