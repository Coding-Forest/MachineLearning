# Set seed for reproducibility
np.random.seed(42)

# Generate an ndarray of integers using: 
# 1) Numpy uniform
# np.random.uniform(start, end, size) + round()

num_ndarray = np.random.uniform(0, 15, 10).round()
# output : array([ 6., 14., 11.,  9.,  2.,  2.,  1., 13.,  9., 11.])
