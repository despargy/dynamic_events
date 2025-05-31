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
    
    def clear_magnitude(self):
        # Init the magnitute
        self.magnitute = np.zeros(self.X.shape)

# Random number to display the center of KG
def C_displ(unit=1):
    return unit*np.random.uniform(-1,1)

if __name__ == "__main__":

    # KG 1
    C1x = 2
    C1y = 5
    STD1=1.3
    # KG 2
    C2x = -3
    C2y = 0
    STD2 = 1.5
    # KG 3
    C3x = 6
    C3y = 1
    STD3 = 1.1
    # Param for KGs' center offset : 1 means +-1
    unit = 2

    Event = ProbEvent()
    for t in range(2):

        K1 = KG((C1x+C_displ(unit), C1y+C_displ(unit)), STD1)
        K2 = KG((C2x+C_displ(unit), C2y+C_displ(unit)), STD2)
        K3 = KG((C3x+C_displ(unit), C3y+C_displ(unit)), STD3)

        kernels_list = [K1, K2, K3] #
        Event.clear_magnitude()
        for K in kernels_list:
            Event.update(K.mv)


        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Event.X, Event.Y, Event.magnitute, cmap='plasma', edgecolor='none')
        ax.set_title(f'T{t+1}: 2D Event representation')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Probability Density')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        # plt.close()

    plt.waitforbuttonpress()
    plt.close('all')