import numpy as np

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    res = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is None:
        return res.squeeze()
    return res

def forward(pi, A, B, O):
  N = A.shape[0]
  T = O.shape[0]
  alpha = np.zeros((T, N))
  
  alpha[0] = pi * B[:,O[0]]
  
  for t in range(1,T):
    alpha[t] = (alpha[t - 1] @ A) * B[:, O[t]]
  
  return alpha

def backward(A, B, O):
  N = A.shape[0]
  T = O.shape[0]
  beta = np.ones((T, N))
  
  for t in reversed(range(T - 1)):
    beta[t] = np.matmul(A , B[:, O[t + 1]] * beta[t + 1] )
  
  return beta

def viterbi(pi, A, B, O):
  N = A.shape[0]
  T = O.shape[0]
  delta = np.zeros((T, N))
  phi = np.zeros((T,N))

  delta[0] = pi * B[:, O[0]]
  for t in range(1,T):
    F_t = delta[t - 1] * A.T
    phi[t] = np.argmax(F_t, axis = 1)
    F_t = np.max(F_t, axis = 1)
    
    delta[t] = F_t * B[:, O[t]]
  
  P = np.max(delta[T - 1])
  
  q = np.zeros(T, dtype = np.int32)
  q[T - 1] = np.argmax(delta[T - 1])
  for t in reversed(range(T - 1)):
    q[t] = phi[t + 1, q[t + 1]]
  
  return P, q