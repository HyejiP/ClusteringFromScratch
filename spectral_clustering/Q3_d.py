from scipy import io, ndimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ll
import math

# First, load the data
mat_file = io.loadmat('isomap.mat')
data = mat_file['images']

######## Below are helper funcitons that are mainly defined in Q2 for PCA ########

def centering(matrix):
    mu = np.mean(matrix, axis=0)
    centered_matrix = matrix - mu[None, :]
    return centered_matrix, mu.reshape(-1, 1)

# compose a covariance matrix
def eigdecomposition(centered_matrix, k):
    C = np.dot(centered_matrix, centered_matrix.T) / width
    eigvals, eigvecs = ll.eigs(C, k=k)
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    return eigvals, eigvecs

# plot k eigenfaces with below function
def print_eigfaces(centered_matrix, eigvals, eigvecs, k, title):
    fig, axs = plt.subplots(1, k, figsize=(2*k, 2))
    for i in range(k):
        dim = (np.dot(eigvecs[:, i].T, centered_matrix)/math.sqrt(eigvals[i])).reshape(width, height)
        dim = ndimage.rotate(dim, -90, reshape=False)
        axs[i].imshow(dim, cmap='gray_r')
        axs[i].axis('off')
        axs[i].set_title(title + str(i))
    return axs

def plot_2D(z, data, title):
    m = z.shape[0]
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # show 20 random images on the plot
    x_size = (max(z[0]) - min(z[0])) * 0.8
    y_size = (max(z[1]) - min(z[1])) * 0.3
    for i in range(20):
        idx = np.random.randint(0, m)
        y0 = z[idx, 1] - (y_size / 2.)
        x1 = z[idx, 0] + (x_size / 2.)
        x0 = z[idx, 0] - (x_size / 2.)
        y1 = z[idx, 1] + (y_size / 2.)
        img = data[:, idx].reshape(64, 64)
        img = ndimage.rotate(img, -90, reshape=False)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
                interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # show 2D scatter plot 
    ax.scatter(z[:,0], z[:,1])
    return ax


##################################################################################

width, height = 64, 64

# transpose the data file so that we can use fat and short matrix (each row represents one image)
# this way eigendecomposition is much faster and takes up less memory
data_t = data.T
centered_data, mu = centering(data_t)

K = 2
eigvals, eigvecs = eigdecomposition(centered_data, k=K)

axs = print_eigfaces(centered_data, eigvals, eigvecs, k=K, title='Eigenface ')
# axs.set_title('Top 2 Eigenfaces', loc='center')
plt.show()

# low dimensional representation
Z_T = eigvecs @ np.sqrt(np.diag(eigvals))

ax = plot_2D(z=Z_T, data=data, title='Two Dimensional Embedding - PCA')
plt.show()

######################################### END ####################################