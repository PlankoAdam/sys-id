import numpy as np
import sys
import os

if __name__ == "__main__":
    try:
        fpath = sys.argv[1]
        fpath = os.path.join(fpath)
    except IndexError:
        quit()
    data = np.load(fpath)
    print(data)