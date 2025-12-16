##AMITH NANJESH##

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, kaiser
from ISCD_lab import ISCD_AnalogDiscovery3

# Function to calculate SNR (Signal-to-Noise Ratio) using DFT
def calculate_SNR_DFT(data, window_function, sampling_frequency):
    N = len(data)
    windowed_data = data * window_function
    dft_data = np.fft.fft(windowed_data)

    # Handle odd and even number of points
    if N % 2 > 0:  # odd number of points FFT
        dft_data[1:len(dft_data)] = dft_data[1:len(dft_data)]
    else:  # even number of points FFT
        dft_data[1:len(dft_data) - 1] = dft_data[1:len(dft_data) - 1]

    # get sampling frequency from time points and create frequency array
    freq = np.fft.fftfreq(N, d=1/sampling_frequency)

    # Calculate SNR using DFT
    fundamental_index = int(N / 10)
    noise_power = np.mean(np.abs(dft_data[int(9 * N / 10):])**2)
    signal_power = np.max(np.abs(dft_data))**2
    snr = 10 * np.log10(signal_power / noise_power)

    return freq[:N//2], np.abs(dft_data[:N//2]), snr

# Function to calculate SINAD (Signal-to-Noise and Distortion Ratio) using DFT
def calculate_SINAD_DFT(data, window_function, sampling_frequency):
    freq, dft_data, SNR = calculate_SNR_DFT(data, window_function, sampling_frequency)
    THD = calculate_THD_DFT(data, window_function, sampling_frequency)
    return freq, dft_data, SNR - THD

# Function to calculate THD (Total Harmonic Distortion) using DFT
def calculate_THD_DFT(data, window_function, sampling_frequency):
    N = len(data)
    windowed_data = data * window_function
    dft_data = np.fft.fft(windowed_data)

    # Find the fundamental frequency index
    fundamental_index = int(N / 10)

    # Extract fundamental frequency power
    fundamental_power = np.abs(dft_data[fundamental_index])**2

    # Extract harmonic powers with index wrapping
    harmonic_powers = [np.abs(dft_data[(k+1)*fundamental_index % N])**2 for k in range(1, 20)]  # Increase to 20 harmonics

    # Calculate distortion power as the sum of harmonic powers
    distortion_power = np.sum(harmonic_powers)

    return 10 * np.log10(distortion_power / fundamental_power)

# Function to calculate ENOB (Effective Number of Bits) using SNR
def calculate_ENOB(SNR):
    return (SNR - 1.76) / 6.02

# Function to calculate SFDR (Spurious-Free Dynamic Range) using DFT
def calculate_SFDR_DFT(data, window_function, sampling_frequency):
    N = len(data)
    windowed_data = data * window_function
    dft_data = np.fft.fft(windowed_data)
    fundamental_power = np.abs(dft_data[int(N / 10)])**2
    max_spurious_power = np.max(np.abs(dft_data)**2) - fundamental_power
    return 10 * np.log10(fundamental_power / max_spurious_power)

# Open device (True means "mocking" virtual device)
AD3 = ISCD_AnalogDiscovery3(True)

# Enable power supply and wait for power-up
AD3.pwrOn()
time.sleep(5)

# Manually set frequency and sampling frequency
chosen_freq = 3000  # Replace with your desired frequency in Hz
chosen_sampling_frequency = 24414  # Replace with your desired sampling frequency in Hz

# Ensure Nyquist criteria is satisfied
if chosen_freq >= chosen_sampling_frequency / 2:
    raise ValueError("Nyquist criteria not satisfied. Increase signal duration or decrease desired resolution.")

# Stimulus creation
print(f"Generating sine wave with chosen frequency: {chosen_freq} Hz")
AD3.awgSIN(0.5, 0.75, chosen_freq)

# Analog data acquisition
print("Read ADC data (synchronous with clkout)...")
data = AD3.laReadADC(1000000)

# extract lower 10 bit and use the correct sign given by the sign bit
adc = [-1 * (i & ((1 << 10) - 1)) if ((i >> 10) & 1) else i & ((1 << 10) - 1) for i in data]

# Choose Kaiser window with beta parameter 14 for DFT
beta = 14
window_function = get_window(('kaiser', beta), len(adc))

# Calculate dynamic parameters using DFT with Kaiser window
dft_freq, dft_data, SNR = calculate_SNR_DFT(adc, window_function, chosen_sampling_frequency)
ENOB = calculate_ENOB(SNR)
SINAD = calculate_SINAD_DFT(adc, window_function, chosen_sampling_frequency)
THD = calculate_THD_DFT(adc, window_function, chosen_sampling_frequency)
SFDR = calculate_SFDR_DFT(adc, window_function, chosen_sampling_frequency)
print(f"Effective number of bits: {ENOB}")

# Display DFT information
print("\nDFT Information:")
print(f" - Number of samples: {len(adc)}")
print(f" - Sampling frequency: {chosen_sampling_frequency}")
print(f" - Max frequency: {chosen_sampling_frequency / 2} Hz")  # Nyquist frequency

# Show results
fig, axs = plt.subplots(4, 1, figsize=(15, 20))

# ADC Data Plot in LSB
axs[0].plot(adc)
axs[0].set_ylabel("ADC data [LSB]")
axs[0].set_xlabel("samples")
axs[0].grid(True)

# Frequency Plot
axs[1].plot(dft_freq, 20*np.log10(np.abs(dft_data)))
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("Magnitude [dB]")
axs[1].grid(True)

# Display dynamic parameters as a table
table_data_dynamic = [
    ['Parameter', 'Value'],
    ['SNR', f'{SNR:.2f} dB'],
    ['SINAD', f'{SINAD[2]:.2f} dB'],
    ['THD', f'{THD:.2f} dB'],
    ['ENOB', f'{ENOB:.2f} bits'],
    ['SFDR', f'{SFDR:.2f} dB']
]

table_dynamic = axs[2].table(cellText=table_data_dynamic, loc='center')
table_dynamic.auto_set_font_size(False)
table_dynamic.set_fontsize(10)
axs[2].axis('off')

# DFT Plot with Harmonics
axs[3].plot(dft_freq, -1*20*np.log10(np.abs(dft_data)))
axs[3].set_xlabel("Frequency [Hz]")
axs[3].set_ylabel("Magnitude [dB]")
axs[3].grid(True)
# Invert the y-axis
axs[3].invert_yaxis()
# Set x-axis limits for the harmonic plot
axs[3].set_xlim([0, chosen_freq * 4])

plt.show()

# Switch off everything
AD3.pwrOff()
del AD3
