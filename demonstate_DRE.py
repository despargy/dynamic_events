import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


class KG:

    def __init__(self, c, std):

        self.cent = c
        self.std = std
        self.cov = np.eye(2)*std**2
        self.mv = multivariate_normal(mean=self.cent, cov=self.cov)

class ProbEvent:

    def __init__(self):
        
        # Create mesh
        x = np.linspace(-10, 10, 200) # Maybe define as the size of the input frame
        y = np.linspace(-10, 10, 200) # Maybe define as the size of the input frame
        self.X, self.Y = np.meshgrid(x, y)
        # Produce the 'terrain'
        self.terrain = np.dstack((self.X, self.Y))
        # Init the magnitute
        self.magnitute = np.zeros(self.X.shape)

    def update(self, mv):
        self.magnitute += mv.pdf(self.terrain)

if __name__ == "__main__":

    Event = ProbEvent()
    K1 = KG((2,5), 1.3)
    K2 = KG((-3,0),1.5)
    K3 = KG((6,1), 1.1)

    kernels_list = [K1, K2, K3] #
    for K in kernels_list:
        Event.update(K.mv)


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Event.X, Event.Y, Event.magnitute, cmap='plasma', edgecolor='none')
    ax.set_title('2D Event representation')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Probability Density')
    plt.tight_layout()
    plt.show()