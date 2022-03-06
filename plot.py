import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_excel("./J00170.xlsx")
    y1 = data.iloc[7500:10001,2]
    y2 = data.iloc[7500:10001,3]
    print(y2)
    plt.plot(range(7500,10001),list(y2))
    plt.plot(range(7500,10001), list(y1))
    plt.legend(["True curve","prediction curve"])
    plt.show()