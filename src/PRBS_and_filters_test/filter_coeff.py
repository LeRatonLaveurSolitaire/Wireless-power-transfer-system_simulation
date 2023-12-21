import numpy as np
import matplotlib.pyplot as plt

def main():
    K =1
    T_s = 1e-6
    f_s = 68000
    Z1 = np.exp(-2*np.pi*T_s*f_s)

    print(Z1)

    N = 12

    bs = []
    cs = []

    for n in range(N):
        b = K * (n==0) - K*(Z1**n)  # n-th coeiffient of the pre-emphasis filter 
        bs.append(b)

        c = 1/K * (Z1**n)           # n-th coeiffient of the de-emphasis filter
        cs.append(c)

    plt.plot([n for n in range(N)],bs,'x')
    plt.title("pre-emphasis filter coefficient")
    #plt.show()

    plt.plot([n for n in range(N)],cs,'x')
    plt.title("de-emphasis filter coefficient")
    #plt.show()



if __name__ == "__main__":
    main()