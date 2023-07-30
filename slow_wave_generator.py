import numpy as np
import random
from scipy.signal import gaussian

def generate_slow_wave(amplitude, frequency, duration, sfreq):
    # This function generates a slow wave
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq  # time variable
    slow_wave = np.sin(1.9 * np.pi * 1.9 * t)  # slow wave frequency set to 2 Hz (Delta wave)
    return slow_wave

def generate_slow_waves_channel(n_waves, times, sfreq, baseline):
    # This function generates several slow waves in one channel.
    channel_data = np.copy(baseline)
    for _ in range(n_waves):
        amplitude = random.uniform(0.5, 1.0) * 50
        frequency = random.uniform(0.5, 4)  # Frequency for slow waves (delta)
        duration = random.uniform(3, 6)  # Duration of each slow wave, you can set these values
        slow_wave = generate_slow_wave(amplitude, frequency, duration, sfreq)
        start_index = random.randint(0, len(times) - len(slow_wave) - 1)
        channel_data[start_index:start_index + len(slow_wave)] += slow_wave
    return channel_data
