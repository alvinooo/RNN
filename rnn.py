import numpy as np

x_dim = 256
h_dim = 100
y_dim = 256

def add_bias(m, axis=0):
	return np.insert(m, 0, 1, axis=axis)

def to_categorical(string):
	seq = np.array([ord(char) for char in string])
	one_hot = np.zeros((len(seq), 256))
	one_hot[np.arange(len(seq)), seq] = 1
	return one_hot

def tanh(a, deriv=False):
    if deriv:
        return 1 - np.tanh(a) ** 2
    else:
        return np.tanh(a)

def softmax(a):
    return np.exp(a) / np.sum(np.exp(a), axis=0)

# Stack of size input (word/sentence)
# Store activation, input, output
# Backprop through time

def forward_propagation(x, w_xh, w_hh, w_hy):
    T = len(x)
    s = np.zeros((T, h_dim))
    # s[-1] = np.zeros(h_dim)
    # o = np.zeros((T, y_dim))
    # sum_xh = x.dot(w_xh).T
    x_activation = x.dot(w_xh)
    prev = np.zeros(h_dim)
    for t in np.arange(T):
    	h_activation = prev.dot(w_hh)
        s[t] = np.tanh(x_activation[t] + prev.dot(w_hh))
        # Extra Credit
        # s[t] = np.tanh(w_xh[:,x[t]] + w_hh[0].dot(s[t-1]))
        # o[t] = np.insert(s[t], 0, 1).dot(w_hy)
    # print s.shape
    # print np.insert(s, 0, 1, axis=1).shape
    o = softmax(s.dot(w_hy).T).T
    # print o.shape
    return o, s

# U = w_xh
# W = w_hh
# V = w_hy

def debug(shapes):
	print
	for i in shapes:
		print i[0], i[1].shape
	print

# Returns gradients for input, hidden, output weights
def bptt(seq, alpha=0.001):
    seq_len = len(seq) - 1

    # One hot encode
    x = to_categorical(seq[:seq_len])
    t = to_categorical(seq[1:])

    # Weight initialization
    w_xh_range = 1.0 / np.sqrt(x_dim)
    w_hh_range = 1.0 / np.sqrt(h_dim)
    w_hy_range = 1.0 / np.sqrt(h_dim)
    w_xh = np.random.uniform(-w_xh_range, w_xh_range, size=(x_dim + 1, h_dim))
    w_hh = np.random.uniform(-w_hh_range, w_hh_range, size=(h_dim + 1, h_dim))
    w_hy = np.random.uniform(-w_hy_range, w_hy_range, size=(h_dim + 1, y_dim))

    w_xh = np.zeros((x_dim + 1, h_dim))
    w_hh = np.zeros((h_dim, h_dim))
    w_hy = np.zeros((h_dim, y_dim))

    # Outputs and activations
    y, a = forward_propagation(add_bias(x, axis=1), w_xh, w_hh, w_hy)
    # Extra credit
    # y, s = forward_propagation(x, w_xh, [w_hh], w_hy)

    debug([("x", x), ("t", t), ("y", y)])

    # Output
    delta_hy = (t - y)
    x_hy = a.T
    dE_dhy = -(x_hy.dot(delta_hy))

    debug([("delta_hy", delta_hy), ("x_hy", x_hy), ("dE_dhy", dE_dhy), ("w_hy", w_hy)])

    # Hidden
    derivatives = 1 - a ** 2

    print "derivatives", derivatives.shape
    print

    delta_hh = derivatives * delta_hy.dot(w_hy.T)
    for t in np.arange(1, seq_len)[::-1]:
    	delta_hh[t - 1] += derivatives[t] * delta_hh[t].dot(w_hh)
    x_hh = np.insert(a[:seq_len - 1], 0, 0, axis=0).T
    dE_dhh = -(x_hh.dot(delta_hh))

    debug([("delta_hh", delta_hh), ("x_hh", x_hh), ("dE_dhh", dE_dhh), ("w_hh", w_hh)])

    # Input
    # delta = (1 - s[activations] ** 2) * sum(w_xh * delta from hidden)
    # gradient -alpha(delta) * s[inputs]
    delta_xh = delta_hh.dot(w_xh.T)
    x_xh = np.insert(x, 0, 1, axis=1)

    debug([("delta_xh", delta_xh), ("x_xh", x_xh), ("w_hx", w_xh)])

    # We accumulate the gradients in these variables
    # dLdU = np.zeros(self.U.shape)
    # dLdV = np.zeros(self.V.shape)
    # dLdW = np.zeros(self.W.shape)
    # delta_o = o
    # delta_o[np.arange(len(y)), y] -= 1.
    # # For each output backwards...
    # for t in np.arange(T)[::-1]:
    #     dLdV += np.outer(delta_o[t], s[t].T)
    #     # Initial delta calculation: dL/dz
    #     delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
    #     # Backpropagation through time (for at most self.bptt_truncate steps)
    #     for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
    #         # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
    #         # Add to gradients at each previous step
    #         dLdW += np.outer(delta_t, s[bptt_step-1])
    #         dLdU[:,x[bptt_step]] += delta_t
    #         # Update delta for next step dL/dz at t-1
    #         delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    # return [dLdU, dLdV, dLdW / T] # Average or sum?

bptt("hello")

# def train(training, test, epochs, h_dim=256, seq_len=10):
# f.read(seq_len)
# with open(filename) as f:
#   while True:
#     c = f.read(1)
#     if not c:
#       print "End of file"
#       break
#     print "Read a character:", c