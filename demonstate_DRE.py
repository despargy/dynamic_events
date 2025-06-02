import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2

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
    
    def __init__(self, c, std):

        self.cent = c
        self.cov = std*np.eye(2)
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

        #TODO remove this - START
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(self.X, self.Y, self.magnitute, cmap='plasma', edgecolor='none')
        # ax.set_title(f'T{t+1}: 2D Event representation WITHOUT NOISE')
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Probability Density')
        # plt.tight_layout()
        #TODO remove this - END

        if noise_samples is not None:
            for n in noise_samples:
                self.magnitute += n.mv.pdf(self.terrain)

    def clear_magnitude(self):
        # Init the magnitute
        self.magnitute = np.zeros(self.X.shape)

# # Random number to display the center of KG
# def C_recenter(unit=1, center=(0,0), cov=np.eye(2)):
#     return unit*np.random.multivariate_normal(mean=center,cov=cov)


class Kernel_Handler:

    core_K : Kernel_Instance
    current_K : Kernel_Instance

    def __init__(self, cx, cy, std):

        self.core_K = Kernel_Instance((cx,cy),std)

    def update_current_K(self):
        
        #TODO check this change os recentering
        # self.current_K = Kernel_Instance(C_recenter(unit=UNIT_RECENT, center=self.core_K.cent, cov=self.core_K.cov ), self.core_K.std)
        new_center = truncated_gaussian_2d(mean=self.core_K.cent, cov=self.core_K.cov, size=1, keep_ratio=0.1)[0]
        self.current_K = Kernel_Instance(c=new_center, std=self.core_K.std)

    def create_noise(self):
        n_noise = np.random.randint(0,MAX_N_NOISE+1)

        #TODO change this - not correct why - add 60% e.g.
        # self.center_noise_samples = np.random.multivariate_normal(mean=self.current_K.cent, cov=self.current_K.cov, size=n_noise)
        self.center_noise_samples = truncated_gaussian_2d(mean=self.current_K.cent, cov=self.current_K.cov, size=n_noise, keep_ratio=0.6) 
        self.noise_samples = []
        for c in self.center_noise_samples:
            self.noise_samples.append( Noise(c=c, std=0.6) ) # random small  #TODO define std


def truncated_gaussian_2d(mean, cov, size, keep_ratio=0.6, oversample_factor=2):
    """
    Sample from a 2D multivariate Gaussian, keeping only samples
    within the central `keep_ratio`% of the probability mass.
    
    Parameters:
        mean: array-like (2,) - center of the Gaussian
        cov: array-like (2, 2) - covariance matrix
        size: int - number of final samples to return
        keep_ratio: float - between 0 and 1, e.g. 0.6 for 60%
        oversample_factor: int - how much extra to sample initially

    Returns:
        ndarray of shape (size, 2) - accepted samples
    """
    dim = 2
    n_total = size * oversample_factor

    # Step 1: Sample extra points
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=n_total)

    # Step 2: Compute Mahalanobis distance for each point
    diffs = samples - mean
    inv_cov = np.linalg.inv(cov)
    mahal_dists = np.einsum('...i,ij,...j->...', diffs, inv_cov, diffs)

    # Step 3: Get the chi-squared threshold for the desired probability mass
    chi2_threshold = chi2.ppf(keep_ratio, df=dim)

    # Step 4: Filter points inside the ellipse
    accepted_samples = samples[mahal_dists <= chi2_threshold]

    # Step 5: Retry or truncate if needed
    if len(accepted_samples) < size:
        # Retry with more samples if not enough
        return truncated_gaussian_2d(mean, cov, size, keep_ratio, oversample_factor * 2)
    else:
        return accepted_samples[:size]

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
        # plt.show()
        plt.show(block=False)
        plt.pause(1) 
        plt.close() # UnComment this to keep one plot only

    plt.waitforbuttonpress()
    plt.close('all')