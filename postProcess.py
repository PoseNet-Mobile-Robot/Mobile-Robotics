from matplotlib import pyplot as plt
import numpy as np
import pickle


def main(filePath):

    with open(filePath, 'rb') as f:
        estPose = pickle.load(f)

    plt.figure(1)

    for i in range(len(estPose)-1):
        plt.plot(estPose[i:i+1, 1], estPose[i:i+1, 2])
        plt.pause(0.05)

plt.show()


if __name__ == '__main__':
    main()
