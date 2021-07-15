import numpy as np
import math

# Create decimals between 0 and 1 with step 0.1
nums = np.arange(0, 1.1, 0.1)


# Function calculating the efficacy of a drug. 
# weight = the multiplying constant
# bias = added value

def efficacy(array):
  layer1_vals = []

  # layer 1
  for i in range(0, len(array)):
    # step1. weighting
    weighted = (array[i] * -34.4) + 2.14 
    # step2. softplus function
    softplus = math.log(1 + math.e** weighted)
    # step3. scaling by -1.30
    scaled = softplus*-1.30
    layer1_vals.append(scaled)

  # hidden layer
  hidden_vals = []
  
  for i in range(0, len(array)):
    # step1. weighting
    weighted = (array[i]*-2.52) + 1.29
    # step2. softplus function
    logged = math.log(1+math.e**weighted)  
    # step3. scaling by 2.28
    scaled = logged*2.28
    hidden_vals.append(scaled)

  # efficacy
  effi_vals = []
  
  # Merging the two graphs - add the values of the layer1 and hidden layer in each position. 
  for i in range(0, len(array)):
    sum = layer1_vals[i] + hidden_vals[i] - 0.58
    effi_vals.append(sum)

  return effi_vals

efficacy = efficacy(nums)

# Plot the efficacy of the new drug. 
import matplotlib.pyplot as plt

def plot_efficacy(): 
  plt.plot(nums, efficacy)
  plt.xlabel('dosage')
  plt.ylabel('efficacy')
  plt.show()

plot_efficacy()
