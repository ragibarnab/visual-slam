import numpy as np


pts = np.random.rand(100, 3)


for x in np.nditer(pts):
    print(x)
