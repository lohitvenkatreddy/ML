# A12.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("train (3).csv")
    X = df.drop(columns=["failure"]).values
    y = df["failure"].values

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)

    clf = MLPClassifier(hidden_layer_sizes=(64,32,16), activation='relu',
                        solver='adam', learning_rate_init=0.001,
                        max_iter=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print("Train accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test accuracy:", accuracy_score(y_test, y_pred_test))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred_test))
    print("\nClassification report (test):\n", classification_report(y_test, y_pred_test))

    # Plot loss curve
    plt.plot(clf.loss_curve_)
    plt.title("MLPClassifier Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
