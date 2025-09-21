# A9.py
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(s): return s*(1-s)

def perceptron_single_layer_lr(X, y, lr=0.05, max_epochs=100):
    # simple online perceptron with sigmoid update (will typically fail for XOR)
    n_features = X.shape[1]
    W = np.random.randn(n_features+1,1)*0.01
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(X)):
            xi = X[i:i+1]
            net = W[0] + xi @ W[1:]
            out = sigmoid(net).item()
            err = y[i] - out
            total_error += err**2
            grad = err * out * (1-out)
            W[1:] += lr * grad * xi.T
            W[0]  += lr * grad
    preds = (sigmoid(np.hstack([np.ones((X.shape[0],1)), X]) @ W) >= 0.5).astype(int).flatten()
    return preds, W

def backprop_XOR(X, y, lr=0.05, hidden_neurons=2, tol=0.002, max_epochs=1000):
    np.random.seed(1)
    W1 = np.random.randn(2, hidden_neurons)*0.1
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons,1)*0.1
    b2 = np.zeros((1,1))
    for epoch in range(1, max_epochs+1):
        total_error=0
        for i in range(len(X)):
            xi = X[i:i+1]
            t = y[i:i+1]
            z1 = xi @ W1 + b1; a1 = sigmoid(z1)
            z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
            err = t - a2; total_error += (err**2).sum()
            delta2 = err * sigmoid_deriv(a2)
            dW2 = a1.T @ delta2; db2 = delta2
            delta1 = (delta2 @ W2.T) * sigmoid_deriv(a1)
            dW1 = xi.T @ delta1; db1 = delta1
            W2 += lr * dW2; b2 += lr * db2
            W1 += lr * dW1; b1 += lr * db1
        if total_error <= tol:
            print("Converged at epoch", epoch); break
    z1 = X @ W1 + b1; a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
    preds = (a2 >= 0.5).astype(int).flatten()
    return preds

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor = np.array([[0],[1],[1],[0]], dtype=float)

    # single-layer attempt (should fail)
    preds_single, W = perceptron_single_layer_lr(X, y_xor, lr=0.05, max_epochs=500)
    print("Single-layer perceptron predictions (XOR) -> should not match perfectly:", preds_single)

    # backprop network (should learn XOR)
    preds_bp = backprop_XOR(X, y_xor, lr=0.05, hidden_neurons=2, tol=0.002, max_epochs=1000)
    print("Backprop predictions (XOR):", preds_bp)
