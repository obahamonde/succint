import time

import numpy as np

start = time.time()
for i in range(1000):
    print(np.random.randint(0, 10))
end = time.time()
print(end - start)
