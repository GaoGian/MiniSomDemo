from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('./resource/house.jpg')
plt.imshow(img)
print(img.shape)

pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))
print(pixels.shape)

som = MiniSom(x= 3, y = 3, input_len = 3, sigma=0.1, learning_rate=0.2)
som.random_weights_init(pixels)
starting_weights = som.get_weights().copy()
som.train_random(pixels, 100)

qnt = som.quantization(pixels)

clustered = np.zeros(img.shape)

for i, q in enumerate(qnt):
  clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q

plt.figure(figsize=(12, 6))

plt.subplot(221)
plt.title('Original')
plt.imshow(img)
plt.subplot(222)
plt.title('Result')
plt.imshow(clustered)

plt.subplot(223)
plt.title('Initial Colors')
plt.imshow(starting_weights)
plt.subplot(224)
plt.title('Learnt Colors')
plt.imshow(som.get_weights())

plt.tight_layout()
plt.show()
