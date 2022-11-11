from scipy import io, ndimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.graph_shortest_path import graph_shortest_path
import scipy.sparse.linalg as ll
from scipy.spatial.distance import cityblock

# load the data
mat_file = io.loadmat('isomap.mat')
data = mat_file['images']

################## Below are the helper functions #####################################

# compose weighted adjacency matrix using Manhattan distance
def get_adj_matrix_manhattan(data, eps):
    m = data.shape[1] # m = number of images
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            # this package was faster than manually implementing 'sum(abs(data[:, i] - data[:, j]))'
            dist = cityblock(data[:,i].reshape(1,-1), data[:,j].reshape(1,-1)) 
            if dist <= eps:
                A[i, j] = dist
    return A

# compute low dimensional representation 
def get_z(adj_matrix, k):
    m = adj_matrix.shape[0]
    D = graph_shortest_path(A)
    H = np.eye(m) - ((np.ones(m).reshape(-1,1) @ np.ones(m).reshape(1,-1)) / m)
    C = -1/2 * (H @ (D * D) @ H)

    eigvals, eigvecs = ll.eigs(C, k=k)
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    
    Z_T = eigvecs @ np.sqrt(np.diag(eigvals))

    return Z_T

# Below function helps to plot
def plot_2D(z, data, title):
    m = z.shape[0]
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # show 20 random images on the plot
    x_size = (max(z[0]) - min(z[0])) * 0.2
    y_size = (max(z[1]) - min(z[1])) * 0.2
    for i in range(20):
        idx = np.random.randint(0, m)
        y0 = z[idx, 1] - (y_size / 2.)
        x1 = z[idx, 0] + (x_size / 2.)
        x0 = z[idx, 0] - (x_size / 2.)
        y1 = z[idx, 1] + (y_size / 2.)
        img = data[:, idx].reshape(64, 64)
        img = ndimage.rotate(img, -90, reshape=False) # rotate the image
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
                interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # show 2D scatter plot 
    ax.scatter(z[:,0], z[:,1])
    return ax


################### Below is code for Q3.3  ######################################

A = get_adj_matrix_manhattan(data=data, eps=500)

fig = plt.figure(figsize=(5, 5)) 
plt.imshow(A, cmap="Greys", interpolation="none")
plt.title('Adjacency Matrix - Manhattan Distance')
plt.show()

Z_T = get_z(A, k=2)

ax = plot_2D(z=Z_T, data=data, title='Two Dimensional Embedding - ISOMAP - Manhattan Distance')
plt.show()

######################################### END ####################################
