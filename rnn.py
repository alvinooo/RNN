import numpy as np

DEBUG = False

x_dim = 256
y_dim = 256

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

# Returns the output and hidden unit activations of the network
def forward_propagation(x, h_dim, w_xh, w_hh, w_hy):
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
	if not DEBUG:
		return
	print
	for i in shapes:
		print i[0], i[1].shape
	print

# Returns gradients for input, hidden, output weights
def bptt(seq, h_dim, w_xh, w_hh, w_hy, alpha=0.001):
    seq_len = len(seq) - 1

    # One hot encode
    x = to_categorical(seq[:seq_len])
    t = to_categorical(seq[1:])

    # Outputs and activations
    y, a = forward_propagation(np.insert(x, 0, 1, axis=1), h_dim, w_xh, w_hh, w_hy)
    debug([("x", x), ("t", t), ("y", y)])
    # Extra credit
    # y, s = forward_propagation(x, w_xh, [w_hh], w_hy)

    derivatives = 1 - a ** 2
    # print "derivatives", derivatives.shape

    # Output
    delta_hy = (t - y)
    x_hy = a.T
    dE_dhy = -(x_hy.dot(delta_hy))
    debug([("delta_hy", delta_hy), ("x_hy", x_hy), ("dE_dhy", dE_dhy), ("w_hy", w_hy)])

    # Hidden
    delta_hh = derivatives * delta_hy.dot(w_hy.T)
    for t in np.arange(1, seq_len)[::-1]:
    	delta_hh[t - 1] += derivatives[t] * delta_hh[t].dot(w_hh)
    x_hh = np.insert(a[:seq_len - 1], 0, 0, axis=0).T
    dE_dhh = -(x_hh.dot(delta_hh))
    debug([("delta_hh", delta_hh), ("x_hh", x_hh), ("dE_dhh", dE_dhh), ("w_hh", w_hh)])

    # Input
    delta_xh = derivatives * delta_hh.dot(w_hh.T)
    x_xh = np.insert(x, 0, 1, axis=1)
    dE_dxh = -(x_xh.T.dot(delta_xh))
    debug([("delta_xh", delta_xh), ("x_xh", x_xh), ("dE_dxh", dE_dxh), ("w_hx", w_xh)])

    return dE_dxh, dE_dhh, dE_dhy

def train(seq, epochs=10, h_dim=100):

	# Weight initialization
    w_xh_range = 1.0 / np.sqrt(x_dim)
    w_hh_range = 1.0 / np.sqrt(h_dim)
    w_hy_range = 1.0 / np.sqrt(h_dim)
    w_xh = np.random.uniform(-w_xh_range, w_xh_range, size=(x_dim + 1, h_dim))
    w_hh = np.random.uniform(-w_hh_range, w_hh_range, size=(h_dim, h_dim))
    w_hy = np.random.uniform(-w_hy_range, w_hy_range, size=(h_dim, y_dim))

    # w_xh = np.zeros((x_dim + 1, h_dim))
    # w_hh = np.zeros((h_dim, h_dim))
    # w_hy = np.zeros((h_dim, y_dim))
    for i in range(epochs):
    	dE_dxh, dE_dhh, dE_dhy = bptt(seq, h_dim, w_xh, w_hh, w_hy)
    	w_xh -= dE_dxh
    	w_hh -= dE_dhh
    	w_hy -= dE_dhy
    	# print dE_dxh
    	# print dE_dhh
    	# print dE_dhy
    return w_xh, w_hh, w_hy

print to_categorical("\00")
# def generate(train, seq_len):
# 	w_xh, w_hh, w_hy = train(train, epochs=2)
# 	forward_propagation("")

# s = "aaaaaaaaaa"
# print generate(s, len(s))
# f.read(seq_len)
# with open(filename) as f:
#   while True:
#     c = f.read(1)
#     if not c:
#       print "End of file"
#       break
#     print "Read a character:", c