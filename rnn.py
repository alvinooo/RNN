x_dim = 256
h_dim = 256
y_dim = 256

def tanh(a, deriv=False):
    if deriv:
        return 1 - np.tanh(a) ** 2
    else:
        return np.tanh(a)

def softmax(a):
    return np.exp(a) / np.sum(np.exp(a), axis=0)

# def forward()

# def train(training, test, epochs, h_dim=256, seq_len=10):