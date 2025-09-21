# A11.py
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def run_mlp_on_table(X, y, hidden_layers=(4,), activation='relu', max_iter=1000):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation,
                        solver='adam', max_iter=max_iter, random_state=1)
    clf.fit(X, y.ravel())
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    return clf, acc, cm

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)

    # AND
    y_and = np.array([[0],[0],[0],[1]])
    clf_and, acc_and, cm_and = run_mlp_on_table(X, y_and, hidden_layers=(4,), activation='logistic', max_iter=500)
    print("AND - MLPClassifier accuracy:", acc_and)
    print("Confusion matrix:\n", cm_and)

    # XOR
    y_xor = np.array([[0],[1],[1],[0]])
    clf_xor, acc_xor, cm_xor = run_mlp_on_table(X, y_xor, hidden_layers=(4,2), activation='logistic', max_iter=2000)
    print("XOR - MLPClassifier accuracy:", acc_xor)
    print("Confusion matrix:\n", cm_xor)
