import numpy as np
import random
from scipy.signal import gaussian

def generate_spike(amplitude, duration, sfreq):
    # This function generates a peak wave
    n_samples = int(duration * sfreq)
    spike = gaussian(n_samples, std=n_samples/7)
    return amplitude * spike / np.max(spike)

def generate_spikes_channel(n_spikes, times, sfreq, baseline):
    # This function generates several point waves in a channel.
    channel_data = np.copy(baseline)
    for _ in range(n_spikes):
        amplitude = random.uniform(0.5, 1.0) * 100
        duration = random.uniform(0.02, 0.07)
        spike = generate_spike(amplitude, duration, sfreq)
        start_index = random.randint(0, len(times) - len(spike) - 1)
        channel_data[start_index:start_index + len(spike)] += spike
    return channel_data