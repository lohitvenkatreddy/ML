import numpy as np
def matrix_power(matrix, exp):
    return np.linalg.matrix_power(matrix, exp)
n = int(input("enter matrix size: "))
elements = []
print(f"enter {n} elements for each row:")
for i in range(n):
    row = []
    for j in range(n):
        val = int(input("enter element: "))
        row.append(val)
    elements.append(row)
if __name__ == "__main__":
    A = np.array(elements)
    m = int(input("enter power m: "))
    res = matrix_power(A, m)
    print(res)
