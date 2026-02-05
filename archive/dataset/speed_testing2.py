import time
import random
import numpy as np

x = list(range(300000))
random.shuffle(x)
x = np.array([(value, i) for i, value in enumerate(x)])

start_time = time.time()

y = sorted(x, key=lambda item: item[0], reverse=False)
print("Sorted in %.3f seconds." % (time.time() - start_time))
print(y[:10])

new_start_time = time.time()
z = np.sort(x, axis=1)
print(z[:10])
print("Numpy sorted in %.3f seconds." % (time.time() - new_start_time))
print(z[0])
print(z[0][0])