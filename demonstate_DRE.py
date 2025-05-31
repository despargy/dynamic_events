import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Class representing Gaussian Kernels
class Kernel_Instance:

    def __init__(self, c, std):

        self.cent = c
        self.std = std
        self.cov = np.eye(2)*std**2
        self.mv = multivariate_normal(mean=self.cent, cov=self.cov)

# Class representing noise as 'small' Gaussian kernels
class Noise:
    noise = None

class ProbEvent:

    def __init__(self, duration = 5):
        
        # duration
        self.duration = duration
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
def C_displ(unit=1, center=(0,0), cov=np.eye(2)):
    return unit*np.random.multivariate_normal(mean=center,cov=cov)
    # return unit*(np.random.uniform(-1,1), np.random.uniform(-1,1))

if __name__ == "__main__":

    # KG 1
    C1x = 2
    C1y = 5
    STD1=1.3
    C1 = (C1x, C1y)
    # KG 2
    C2x = -3
    C2y = 0
    STD2 = 1.5
    C2 = (C2x, C2y)
    # KG 3
    C3x = 6
    C3y = 1
    STD3 = 1.1
    C3 = (C3x, C3y)

    # The main core KGs - After some time this is the core of the fire 
    core_K1 = Kernel_Instance(C1, STD1)
    core_K2 = Kernel_Instance(C2, STD2)
    core_K3 = Kernel_Instance(C3, STD3)

    # Param for KGs' center offset : 1 means +-1
    unit = 1


    # The event
    Event = ProbEvent(duration=2)

    # Over the time t
    for t in range(Event.duration):

        K1 = Kernel_Instance(C1+C_displ(unit=0.05, center=core_K1.cent, cov=core_K1.cov ), STD1)
        K2 = Kernel_Instance(C2+C_displ(unit=0.2, center=core_K2.cent, cov=core_K2.cov ), STD2)
        K3 = Kernel_Instance(C3+C_displ(unit=0.08, center=core_K3.cent, cov=core_K3.cov ), STD3)

        kernels_list = [K1, K2, K3] #
        Event.clear_magnitude()
        for K in kernels_list:
            Event.update(K.mv)

        # Vizualization over time
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
        plt.close() # UnComment this to keep one plot only

    plt.waitforbuttonpress()
    plt.close('all')