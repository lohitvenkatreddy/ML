from collections import Counter
def max_char(s):
    s = ''.join(c.lower() for c in s if c.isalpha())
    if not s:
        return None, 0
    freq = Counter(s)
    ch, cnt = max(freq.items(), key=lambda x: x[1])
    return ch, cnt
if __name__ == "__main__":
    txt = input("Enter a string: ")
    c, n = max_char(txt)
    print("Highest occurring character:", c)
    print("Count:", n)