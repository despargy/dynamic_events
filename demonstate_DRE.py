import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

SIZE_X = 10
SIZE_Y = 10
NPOINTS = 200
UNIT_RECENT = 0.7
MAX_N_NOISE = 20
MAX_NOISE_COV = 8
# Class representing Gaussian Kernels
class Kernel_Instance:

    def __init__(self, c, std):

        self.cent = c
        self.std = std
        self.cov = np.eye(2)*std**2
        self.mv = multivariate_normal(mean=self.cent, cov=self.cov)

# Class representing noise as 'small' Gaussian kernels
class Noise:
    
    def __init__(self, c, cov):

        self.cent = c
        self.cov = cov
        self.mv = multivariate_normal(mean=self.cent, cov=self.cov)

class ProbEvent:

    def __init__(self, duration = 5):
        
        # duration
        self.duration = duration
        # Create mesh
        x = np.linspace(-SIZE_X, SIZE_X, NPOINTS) # Maybe define as the size of the input frame
        y = np.linspace(-SIZE_Y, SIZE_Y, NPOINTS) # Maybe define as the size of the input frame
        self.X, self.Y = np.meshgrid(x, y)
        # Produce the 'terrain'
        self.terrain = np.dstack((self.X, self.Y))
        # Init the magnitute
        self.magnitute = np.zeros(self.X.shape)

    def update(self, k, noise_samples = None):
        self.magnitute += k.mv.pdf(self.terrain)
        if noise_samples is not None:
            for n in noise_samples:
                self.magnitute += n.mv.pdf(self.terrain)

    def clear_magnitude(self):
        # Init the magnitute
        self.magnitute = np.zeros(self.X.shape)

# Random number to display the center of KG
def C_recenter(unit=1, center=(0,0), cov=np.eye(2)):
    return unit*np.random.multivariate_normal(mean=center,cov=cov)


class Kernel_Handler:

    core_K : Kernel_Instance
    current_K : Kernel_Instance

    def __init__(self, cx, cy, std):

        self.core_K = Kernel_Instance((cx,cy),std)

    def update_current_K(self):
        self.current_K = Kernel_Instance(C_recenter(unit=UNIT_RECENT, center=self.core_K.cent, cov=self.core_K.cov ), self.core_K.std)

    def create_noise(self):
        n_noise = np.random.randint(0,MAX_N_NOISE+1)
        self.center_noise_samples = np.random.multivariate_normal(mean=self.current_K.cent, cov=self.current_K.cov, size=n_noise)
        self.noise_samples = []
        for c in self.center_noise_samples:
            self.noise_samples.append( Noise(c=c, cov=np.random.randint(1,MAX_NOISE_COV) *np.eye(2)) ) # random small  #TODO

if __name__ == "__main__":

    KHandler_list = []
    N_KGs = 3

    # Use this to rangom generate kernels
    # for n in range(N_KGs):
    #     KHandler_list.append(Kernel_Handler(random cx, random cy, random std))
    # Use this to 'static' generate kernels
    KHandler_list.append(Kernel_Handler( 2, 5, 1.3)) # Init the main core KGs 
    KHandler_list.append(Kernel_Handler(-3, 0, 1.5)) # Init the main core KGs
    KHandler_list.append(Kernel_Handler( 6, 1, 1.1)) # Init the main core KGs
    # The event
    Event = ProbEvent(duration=3)

    # Over the time t
    for t in range(Event.duration):

        # Clear magnitude to 'enrich' dynamic
        Event.clear_magnitude()
        # For every main Gaussian Kernel (KG)
        for kh in KHandler_list:
            
            # Recenter a little the center
            kh.update_current_K()
            # Create noise
            kh.create_noise()
            # Update the magnitute for each main KG
            Event.update(kh.current_K, kh.noise_samples)


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
        plt.pause(1) 
        plt.close() # UnComment this to keep one plot only

    plt.waitforbuttonpress()
    plt.close('all')