## AMITH NANJESH ####

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from ISCD_lab import ISCD_AnalogDiscovery3

# Function to perform offset calibration
def calibrateOffset(ad3_instance):
    # Ensure no signal is applied (e.g., set voltage to 0V)
    ad3_instance.awgDC(0, 0.75)

    # Read ADC data (1000 samples)
    adc_data = ad3_instance.laReadADC(10000)

    # Extract lower 10 bits and use the correct sign given by the sign bit
    adc_values = [-1 * (i & ((1 << 10) - 1)) if ((i >> 10) & 1) else i & ((1 << 10) - 1) for i in adc_data]

    # Calculate and return the offset
    return -np.average(adc_values)

# Function to perform gain calibration
def calibrateGain(ad3_instance):
    # Apply a known reference voltage (e.g., 1.0V)
    ad3_instance.awgDC(0.5, 0.75)

    # Read ADC data (1000 samples)
    adc_data = ad3_instance.laReadADC(1000)

    # Extract lower 10 bits and use the correct sign given by the sign bit
    adc_values = [-1 * (i & ((1 << 10) - 1)) if ((i >> 10) & 1) else i & ((1 << 10) - 1) for i in adc_data]

    # Calculate and return the gain
    reference_voltage = 0.5  # Set the known reference voltage
    return reference_voltage / np.average(adc_values)

# Device connection and power-up
AD3 = ISCD_AnalogDiscovery3()  # True for virtual device
AD3.status()
AD3.pwrOn()
time.sleep(5)

# Calibration steps
offset_calibration = calibrateOffset(AD3)
gain_calibration = calibrateGain(AD3)

# Print calibration results
print(f"Offset Calibration: {offset_calibration} LSBs")
print(f"Gain Calibration: {gain_calibration} LSB/V")

# Define ramp parameters
reference_voltage = 0.5  # Reference voltage (optional, for error checking)
ADC_resolution = 11  # Bits
lsb_voltage = reference_voltage / (2**ADC_resolution - 1)
lsb_step = lsb_voltage/3 
ramp_max_voltage = 0.49  # Maximum voltage of the ramp

# Calculate number of steps
num_steps = int(ramp_max_voltage / lsb_step) + 1  # Include 0.5V

# Check for reference voltage limitation
if reference_voltage < ramp_max_voltage:
    print("Warning: Desired peak voltage exceeds reference voltage. Capping to reference voltage.")
    num_steps = int(reference_voltage / lsb_step) + 1
    ramp_max_voltage = reference_voltage

# Generate ramp values with numpy.linspace
voltage_steps = np.linspace(-0.49, ramp_max_voltage, num_steps)

# Array to store average ADC values
avg_adc_values = []
avg_adc_values_not_corrected = []

# Loop through ramp steps
for voltage in voltage_steps:
    AD3.awgDC(voltage, 0.75)  # Apply each step of the ramp

    # Read ADC data (1000 samples per step)
    adc_data = AD3.laReadADC(1000)
    # Extract lower 10 bits and use the correct sign given by the sign bit
    adc = [-1 * (i & ((1 << 10) - 1)) if ((i >> 10) & 1) else i & ((1 << 10) - 1) for i in adc_data]
    
    # Calculate average ADC for this step
    avg_adc = np.average(adc)
    avg_adc_values_not_corrected.append(avg_adc)

    # Apply offset and gain correction
    corrected_adc = [(val - offset_calibration) / gain_calibration for val in adc]
    avg_adc_values.append(np.average(corrected_adc))  # Use the corrected ADC values

# Ideal code values at end points (assuming 0V and ramp_max_voltage)
ideal_code_min = 0
ideal_code_max = int(ramp_max_voltage / reference_voltage * (2 ** ADC_resolution - 1))

# Linear fit to ideal transfer function
slope, intercept, _, _, _ = linregress(voltage_steps, avg_adc_values)
ideal_adc_values_fit = [slope * vin + intercept for vin in voltage_steps]

# INL calculation
INL = [actual - ideal for actual, ideal in zip(avg_adc_values, ideal_adc_values_fit)]

# DNL calculation (exclude first step)
DNL = [actual_diff - ideal_diff for actual_diff, ideal_diff in zip(np.diff(avg_adc_values), np.diff(ideal_adc_values_fit))]

# Convert INL and DNL to LSBs
inl_lsb = [val / (2 ** ADC_resolution - 1) for val in INL]
dnl_lsb = [val / (2 ** ADC_resolution - 1) for val in DNL]

# Plot the results
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)  # Applied ramp
ax2 = fig.add_subplot(2, 2, 2)  # ADC data
ax3 = fig.add_subplot(2, 2, 3)  # INL
ax4 = fig.add_subplot(2, 2, 4)  # DNL

# Plot applied ramp
ax1.plot(voltage_steps, label="Applied Ramp")
ax1.set_ylabel("Voltage (Vin) [V]")
ax1.set_title("Applied Ramp")
ax1.grid(True)
plt.xlim(-0.5, 0.5)

# Plot ADC data
ax2.plot(voltage_steps, avg_adc_values_not_corrected, label="ADC Data")
ax2.set_xlabel("Voltage (Vin) [V]")
ax2.set_ylabel("ADC Code (11 bits)")
ax2.set_title("ADC Data")
ax2.grid(True)
# Set Vin axis limits
plt.xlim(-0.5, 0.5)  # Set minimum and maximum values for Vin axis

# Plot INL as subplot
ax3.plot(voltage_steps, inl_lsb)
ax3.set_xlabel("Voltage (Vin) [V]")
ax3.set_ylabel("INL (LSB)")
ax3.set_title("Integral Nonlinearity (LSBs)")
ax3.grid(True)

# Plot DNL as subplot
ax4.plot(voltage_steps[1:], dnl_lsb)  # Exclude first step for DNL
ax4.set_xlabel("Voltage (Vin) [V]")
ax4.set_ylabel("DNL (LSB)")
ax4.set_title("Differential Nonlinearity (DNL)")
ax4.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Save data to CSV
data_to_save = np.column_stack((voltage_steps, avg_adc_values, ideal_adc_values_fit))
np.savetxt("vramp_and_adc_data.csv", data_to_save, delimiter=",", header="Vramp,ADC Code,Ideal ADC Code")

# Switch off everything
AD3.pwrOff()
del AD3


