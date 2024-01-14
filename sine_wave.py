import numpy as np
from scipy.io.wavfile import write

# Parameters for the sine wave
duration = 5.0  # Duration in seconds
frequency = 440.0  # Frequency in Hz (A4 note)
sampling_rate = 44100  # Sampling rate in Hz
amplitude = 32767  # Max amplitude for 16-bit audio (32767)

# Generate the time values
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate the sine wave
wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Convert to 16-bit integers
wave_integers = np.int16(wave)

# Save to a WAV file
filename = 'sine_wave_440Hz.wav'
write(filename, sampling_rate, wave_integers)

import numpy as np
from scipy.io.wavfile import write

# Parameters for the sine wave
duration = 5.0  # Duration in seconds
frequency = 440.0  # Frequency in Hz (A4 note)
sampling_rate = 44100  # Sampling rate in Hz
amplitude = 32767  # Max amplitude for 16-bit audio (32767)

# Generate the time values
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate the sine wave
wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Apply fade-in and fade-out
fade_time = 0.5  # Fade time in seconds
n_fade_samples = int(fade_time * sampling_rate)

# Fade-in
for i in range(n_fade_samples):
    wave[i] *= (i / n_fade_samples)

# Fade-out
for i in range(n_fade_samples):
    wave[-i-1] *= (i / n_fade_samples)

# Convert to 16-bit integers
wave_integers = np.int16(wave)

# Save to a WAV file
filename = 'sine_wave_440Hz_no_clipping.wav'
write(filename, sampling_rate, wave_integers)

import numpy as np
from scipy.io.wavfile import write

# Parameters for the sine wave
duration = 5.0  # Duration in seconds
frequency = 440.0  # Frequency in Hz (A4 note)
sampling_rate = 44100  # Sampling rate in Hz
amplitude = 32767  # Max amplitude for 16-bit audio (32767)

# Generate the time values
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate the sine wave
wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Apply fade-in and fade-out
fade_time = 0.5  # Fade time in seconds
n_fade_samples = int(fade_time * sampling_rate)

# Fade-in (exponential)
fade_in = np.linspace(0, 1, n_fade_samples)**2
wave[:n_fade_samples] *= fade_in

# Fade-out (exponential)
fade_out = np.linspace(1, 0, n_fade_samples)**2
wave[-n_fade_samples:] *= fade_out

# Convert to 16-bit integers
wave_integers = np.int16(wave)

# Save to a WAV file
filename = 'sine_wave_440Hz_smooth_fade.wav'
write(filename, sampling_rate, wave_integers)

import numpy as np
from scipy.io.wavfile import write

# Parameters for the sine wave
duration = 5.0  # Duration in seconds
frequency = 440.0  # Frequency in Hz (A4 note)
sampling_rate = 44100  # Sampling rate in Hz
amplitude = 32767  # Max amplitude for 16-bit audio (32767)

# Ensure that the wave starts at a zero crossing
start_time = 1 / frequency / 4
end_time = start_time + duration

# Generate the time values
t = np.linspace(start_time, end_time, int(sampling_rate * duration), endpoint=False)

# Generate the sine wave
wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Apply fade-in and fade-out
fade_time = 0.5  # Fade time in seconds
n_fade_samples = int(fade_time * sampling_rate)

# Fade-in (exponential)
fade_in = np.linspace(0, 1, n_fade_samples)**2
wave[:n_fade_samples] *= fade_in

# Fade-out (exponential)
fade_out = np.linspace(1, 0, n_fade_samples)**2
wave[-n_fade_samples:] *= fade_out

# Convert to 16-bit integers
wave_integers = np.int16(wave)

# Save to a WAV file
filename = 'sine_wave_440Hz_smooth_start.wav'
write(filename, sampling_rate, wave_integers)

import matplotlib.pyplot as plt

# Plot the wave
plt.figure(figsize=(10, 6))
plt.plot(t, wave)
plt.title("Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
# plt.show()
import matplotlib.pyplot as plt

# Plot the wave
plt.figure(figsize=(10, 6))
plt.plot(t, wave)
plt.title("Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Save the plot to a file
plot_filename = 'sine_wave_plot.png'
plt.savefig(plot_filename)

import matplotlib.pyplot as plt

# Plot the wave, focusing on a smaller segment for clarity
plt.figure(figsize=(10, 6))
plt.plot(t[:1000], wave[:1000])  # Adjust these indices to focus on different parts of the wave
plt.title("Zoomed-In Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Save the plot to a file
plot_filename = 'zoomed_in_sine_wave_plot.png'
plt.savefig(plot_filename)
