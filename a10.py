# A10.py
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(s): return s*(1-s)

def train_two_output_network(X, Y_targets, hidden_neurons=4, lr=0.05, max_epochs=1000, tol=0.002):
    n_in = X.shape[1]
    np.random.seed(0)
    W1 = np.random.randn(n_in, hidden_neurons)*0.1
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, 2)*0.1
    b2 = np.zeros((1,2))
    for epoch in range(1, max_epochs+1):
        total_error = 0.0
        for i in range(len(X)):
            xi = X[i:i+1]; t = Y_targets[i:i+1]
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
            break
    return W1,b1,W2,b2

def evaluate_two_output(X, W1,b1,W2,b2):
    a1 = sigmoid(X @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    preds = np.argmax(a2, axis=1)  # 0 => class0, 1 => class1
    return a2, preds

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)

    # Map 0 -> [1,0], 1 -> [0,1]
    AND_targets = np.array([[1,0],[1,0],[1,0],[0,1]], dtype=float)
    XOR_targets = np.array([[1,0],[0,1],[0,1],[1,0]], dtype=float)  # note: XOR truth maps

    # AND
    W1,b1,W2,b2 = train_two_output_network(X, AND_targets, hidden_neurons=3, lr=0.05, max_epochs=1000, tol=0.002)
    probs_and, preds_and = evaluate_two_output(X, W1,b1,W2,b2)
    print("AND two-output probs:\n", probs_and)
    print("AND mapped preds (0->[1,0],1->[0,1]):", preds_and)

    # XOR
    W1,b1,W2,b2 = train_two_output_network(X, XOR_targets, hidden_neurons=3, lr=0.05, max_epochs=1000, tol=0.002)
    probs_xor, preds_xor = evaluate_two_output(X, W1,b1,W2,b2)
    print("XOR two-output probs:\n", probs_xor)
    print("XOR mapped preds:", preds_xor)
