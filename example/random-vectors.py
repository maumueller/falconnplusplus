import falconnpp
import numpy as np
X = np.random.normal(size=(100, 10000))
index = falconnpp.FalconnPP(10000, 100)
index.build(X)
Y = np.random.normal(size=(100, 1000))
index.search(Y, 1)