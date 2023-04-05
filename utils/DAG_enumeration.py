from math import factorial


def enumation(n):
    if n <= 1:
        return 1
    result = 0
    for k in range(1, n + 1):
        result += ((-1) ** (k - 1)) * choose(k, n) * (2 ** (k * (n - k))) * enumation(n - k)
    return result


def choose(k, n):
    return factorial(n) / (factorial(k) * factorial(n - k))


if __name__ == "__main__":
    print(enumation(8))
    print(enumation(12))
