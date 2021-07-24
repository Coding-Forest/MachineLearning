import math

# ENTROPY
# choose a random object out of the box mixed with orange stars + orange diamonds + blue stars + blue diamonds.
# When the disorder is high (close to 1), it is hard to predict the probability of which one will be picked.

# ---------------------------------------------------------

# Calculate the disorder of all objects (full group).
box_total = 49
orange_total = 25
blue_total = 24
orange_diamonds = 21
blue_diamonds = 21

# Entropy function for binary classification
# total = total number of objects
# target = number of target objects
def entropy(total, target):
  prob_target = target / total
  prob_rest = (total-target) / total

  entropy_target = -(prob_target * math.log2(prob_target))
  entropy_rest = - (prob_rest) * math.log2(prob_rest)

  return entropy_target + entropy_rest
            
# ---------------------------------------------------------

# Compute entropy with parameterised values
  
# Calculate the disorder of orange objects in the orange box.
# Meaning... when you pick an object at random from either of the colour-separated boxes,
# you have a higher probability of your prediction being correct. 
# then choosing from the full box.
full_box_entropy = entropy(box_total, orange_total)           # 0.9997 = a lot of disorder.
orange_box_entropy = entropy(orange_total, orange_diamonds)   # 0.6343 = less disorder than the full group.
blue_box_entropy = entropy(blue_total, blue_diamonds)         # 0.5436 = The chance of predicting correctly is higher in the blue box than in the orange box

print(f'full box entropy: {full_box_entropy}')
print(f'blue box entropy: {blue_box_entropy}')
print(f'orange box entropy: {orange_box_entropy}')


# ---------------------------------------------------------------------------------

# Combined entropy : entropy for a partition (meaning entropy of a sub-population partitioned according to a certain feature (colour in this case))
box_total = orange_total + blue_total
combined_entropy = (orange_total/box_total * orange_box_entropy) + (blue_total / box_total * blue_box_entropy)
# combined entropy = (orange-in-box probability * orange box entropy) + (blue-in-box probability * blue box entropy)

# Information gain
full_box_entropy = entropy(box_total, orange_total)

# Combined entropy for a partition
combined_entropy = 25/49 * orange_box + 24/49 * blue_box
print(f'combined_entropy: {combined_entropy}')    # 0.5899

# Information Gain
info_gain = full_box_entropy - combined_entropy

print(f'Information Gain: {info_gain}')     # 0.4098
