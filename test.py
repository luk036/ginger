import math  # Optionally

import latexify


@latexify.function
def sinc(x):
    if x == 0:
        return 1
    else:
        return math.sin(x) / x


if __name__ == "__main__":
    print(sinc)
