# This is all you need. As simple as!
import numpy as np
import matplotlib.pyplot as plt

# Create a list of input values.
input_vals = np.arange(0, 1.1, 0.1)
input_vals

# ReLU algorithm in action. 
def ReLU(arr):
  # layer 1
  scaled_vals = []
  for i in range(len(arr)):
    scaled = arr[i] * 1.7 - 0.85
    if scaled <= 0:             # ReLU here.
      scaled_vals.append(0)
    else: scaled = scaled * -40.8
    scaled_vals.append(scaled) 
  
  # hidden layer
  scaled_hidden_vals = []
  for i in range(0, len(arr)):
    scaled = arr[i] * 12.6
    if scaled <= 0:             # ReLU here.
      scaled_hidden_vals.append(0)
    else: scaled = scaled * 2.7
    scaled_hidden_vals.append(scaled)
  
  # merge the two layers
  merged_vals = []
  for i in range(0, len(arr)):
    merged = scaled_vals[i] + scaled_hidden_vals[i] - 16
    if merged <= 0:             # ReLU here.
      merged_vals.append(0)
    else: merged_vals.append(merged)
  
  return merged_vals


# Apply the function and check the result
output_vals = ReLU(input_vals)
output_vals

# Plot the ReLUed output and see what it looks like
def plot_ReLU(x, y, xlabel, ylabel):
  plt.plot(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

plot_ReLU(input_vals, output_vals, 'Dosage', 'Output')
