import numpy as np



X = np.array([[0.2, 0.5, 0.3], [0.5, 0.1, 0.4], [0.3, 0.3, 0.4]])
y = np.array([1, 2, 0])

def stable_softmax(X):
    print(np.max(X,axis=-1, keepdims=True))
    exps = np.exp(X - np.max(X,axis=-1, keepdims=True))
    return np.divide(exps,np.sum(exps,axis=-1, keepdims=True))

m = y.shape[0]
p = stable_softmax(X)
print(p)
# We use multidimensional array indexing to extract
# softmax probability of the correct label for each sample.
# Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
log_likelihood = -np.log(p[range(m),y])
print(log_likelihood)
loss = np.sum(log_likelihood) / m

print(" loss " ,loss)