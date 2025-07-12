import random
import statistics

def stats(n, a, b):
    lst = [random.randint(a, b) for _ in range(n)]
    mean = statistics.mean(lst)
    median = statistics.median(lst)
    mode = statistics.mode(lst)
    return lst, mean, median, mode
if __name__ == "__main__":
    nums, m, md, mo = stats(25, 1, 10)
    print("Numbers:", nums)
    print("Mean:", m)
    print("Median:", md)
    print("Mode:", mo)
