# Calculating a feature map using the kernel
# timestamp: https://www.youtube.com/watch?v=HGwBXDKFk9I&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=13&t=268s

def kernel_feature_map(image, kernel_size=(3,3)):
  # kernel data 
  kernel = np.array([[0, 0, 1], 
                     [0, 1, 0], 
                     [1, 0, 0]]).flatten()
  kernel_rows = kernel_size[0]
  kernel_cols = kernel_size[1]

  # image data necessary for vector convolution
  nrows = image.shape[0]
  ncols = image.shape[1]
  move_limit_rows = kernel_size[0] - 1
  move_limit_cols = kernel_size[1] - 1
  move_count_rows = nrows - move_limit_rows
  move_count_cols = ncols - move_limit_cols

  # Create an empty feature map 
  feature_map = []

  # Create an empty array
  extracted_vectors = []

  # Extract image vectors by the kernel size
  for k in range(nrows - move_limit_rows):
    for i in range(ncols - move_limit_cols):  # 6 - 2 = 4 (move count for columns)
      for j in range(k, k + kernel_cols): # 6 - 2 = 4 (move count for rows)
        extracted_vectors.append(image[j][i:i+kernel_rows])
      extracted_vectors = np.array(extracted_vectors).flatten()
      # get the dot product with the kernel
      filtered_val = sum(extracted_vectors * kernel) - 2  # 2 = bias
      feature_map.append(filtered_val)
      extracted_vectors = []
  
  feature_map = np.array(feature_map).reshape(move_count_rows, move_count_cols)

  return feature_map

feature_map = kernel_feature_map(image_o)
feature_map



# --------------------------------------------------------------------------------------------------------------

# Filter the feature map using the ReLU activation function. 
# timestamp: https://youtu.be/HGwBXDKFk9I?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&t=424

def relu_feature_map(feature_map):
  nrows = feature_map.shape[0]
  ncols = feature_map.shape[1]

  for i in range(nrows):  # controls the move in rows
    for j in range( ncols):  # controls the move in columns
      if feature_map[i][j] > 0:
        feature_map[i][j] = 1
      else: feature_map[i][j] = 0
  
  return feature_map

relued_feature_map = relu_feature_map(feature_map)
relued_feature_map



# --------------------------------------------------------------------------------------------------------------

# Further convolute (reduce) the filtered feature map using the MaxPooling 
# MaxPooling pools maximum numbers from each region of a vector.
# timestamp: https://www.youtube.com/watch?v=HGwBXDKFk9I&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=13&t=478s

def maxpool_feature_map(relued_feature_map):
  nrows = relued_feature_map.shape[0]
  ncols = relued_feature_map.shape[1]
  step = int(nrows / 2)

  # Create an empty array to store the Post ReLU extraction
  pre_pool_vals = []
  maxpooled_vals = []

  for i in range(0, nrows, step):
    for j in range(0 + i, step + i):
      pre_pool_vals.append(relued_feature_map[j][:2])
      pre_pool_vals.append(relued_feature_map[j][2:])
      maxpooled_vals.append(max(np.array(pre_pool_vals).flatten()))
      pre_pool_vals = []
  return np.array(maxpooled_vals).reshape(2,2)

maxpooled_vals = maxpool_feature_map(relued_feature_map)
maxpooled_vals



# --------------------------------------------------------------------------------------------------------------

# Neural Network classifies the image by jugding the image summary (convoluted vector).
# timestamp: https://www.youtube.com/watch?v=HGwBXDKFk9I&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=13&t=588s

def classify_xo(maxpooled_vals):
  nrows = maxpooled_vals.shape[0]
  ncols = maxpooled_vals.shape[1]
  weights = np.array([[-0.8, -0.07],
                     [0.2, 0.17]])

  weighted_vals = []
  
  for i in range(nrows):
    for j in range(ncols):
      weighted = maxpooled_vals[i][j] * weights[i][j]
      weighted_vals.append(weighted)
  biased = sum(weighted_vals) + 0.97  # x-axis coordinate for the Activation Function.
  
  # ReLU activation
  if biased > 0:
    biased = biased
  else: biased = 0

  o_classifier = round(biased * -1.33 + 1.45, 2)
  x_classifier = round(biased *  1.33 + -0.45, 2)
  classified_vals = [o_classifier, x_classifier]
  
  if o_classifier == 1:
    return 'o'
  else: return 'x'

classify_xo(maxpooled_vals)



# --------------------------------------------------------------------------------------------------------------

# The combination of all the steps in order.
def classify_xo_image(image):
  feature_map = kernel_feature_map(image, kernel_size=(3,3))
  relued_feature_map = relu_feature_map(feature_map)
  maxpooled_vals = maxpool_feature_map(relued_feature_map)
  classified_result = classify_xo(maxpooled_vals)
  return classified_result



# --------------------------------------------------------------------------------------------------------------

# Test the model by feeding some x/o images.
# timestamp: https://www.youtube.com/watch?v=HGwBXDKFk9I&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=13&t=689s

image_o = np.array([[0, 0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0]])

image_x = np.array([[1, 0, 0, 0, 0, 1], 
                    [0, 1, 0, 0, 1, 0],
                    [0, 0 ,1, 1, 0, 0],
                    [0, 0 ,1, 1, 0, 0],           
                    [0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1]])

# One of the advantages of the CNN is that it's tolerant to shifted images.
image_shifted_o = np.array([[0, 0, 1, 1, 0, 0],
                            [0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0],
                            [0, 0, 1, 1, 0, 0]])

image_shifted_x = np.array([[0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1],
                            [0, 0, 0 ,1, 1, 0],
                            [0, 0, 0 ,1, 1, 0],
                            [0, 0, 1, 0, 0, 1],
                            [0, 1, 0, 0, 0, 0]])
 
print(classify_xo_image(image_o))
print(classify_xo_image(image_x))
classify_xo_image(image_shifted_x)
classify_xo_image(image_shifted_o)
