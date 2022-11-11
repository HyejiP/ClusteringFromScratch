from scipy import io, ndimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.graph_shortest_path import graph_shortest_path
import scipy.sparse.linalg as ll

# load the data
mat_file = io.loadmat('isomap.mat')
data = mat_file['images']

################## Below are the helper functions #####################################

# compose weighted adjacency matrix using Euclidean distance
def get_adj_matrix(data, eps):
    m = data.shape[1] # m = number of images
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dist = np.linalg.norm(data[:,i] - data[:,j]) 
            if dist <= eps:
                A[i, j] = dist
    return A

# compute low dimensional representation
def get_z(adj_matrix, k):
    m = adj_matrix.shape[0] 
    D = graph_shortest_path(A) # construct graph distance matrix D based on geodesic distance
    H = np.eye(m) - ((np.ones(m).reshape(-1,1) @ np.ones(m).reshape(1,-1)) / m) # construct centering matrix H
    C = -1/2 * (H @ (D * D) @ H) 

    eigvals, eigvecs = ll.eigs(C, k=k) # compute k largest(in this case, k=2) eigenvalues and corresponding eigenvectors
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    
    Z_T = eigvecs @ np.sqrt(np.diag(eigvals)) # low dimensional representation

    return D, Z_T

# Below function helps to plot
def plot_2D(z, data, title):
    m = z.shape[0]
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # show 20 random images on the plot
    x_size = (max(z[0]) - min(z[0])) * 0.22
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

################ Below is code for Q3.1 & Q3.2  ##################################

A = get_adj_matrix(data=data, eps=12) # eps=12 is derived from the below tuning epsilon value portion

fig = plt.figure(figsize=(5, 5)) 
plt.imshow(A, cmap="Greys", interpolation="none")
plt.title('Adjacency Matrix - Euclidean Distance')
plt.show()

D, Z_T = get_z(A, k=2)

ax = plot_2D(z=Z_T, data=data, title='Two Dimensional Embedding - ISOMAP - Euclidean Distance')
plt.show()

################### Tuning epsilon value ##########################################

eps_vals = list(range(10,20)) # epsilon values to test ranging 10-19


# below function will help us recover the Euclidean distance matrix from low dimensional representation(2D), Z_T
def recover_dist(z):
    z = z.T
    m = data.shape[1]
    recovered_D = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dist = np.linalg.norm(z[:,i] - z[:,j])
            recovered_D[i, j] = dist
    return recovered_D


# compute correlation coefficient 
# between geodesic distance matrix D and recovered Euclidean distance matrix from low dimensional representation
residuals = [] # this will contain residual variance; (1 - correlation coefficient^2) at each epsilon value
for e in eps_vals:
    print('-tuning epsilon value... epsilon = ', e)
    A = get_adj_matrix(data=data, eps=e)
    D, Z_T = get_z(A, k=2)
    
    recovered_D = recover_dist(Z_T) # recover Euclidean distance between images from 2D respresntation using above function
    
    pearson = np.corrcoef(D.reshape(1,-1), recovered_D.reshape(1,-1)) # compute correlation cofficient 
    
    residuals.append(1 - (pearson[0,1]**2))

plt.plot(eps_vals, residuals) # plot residual variances
plt.xlabel('Epsilon')
plt.ylabel('1 - R^2')
plt.title('Tuning epsilon by computing "1 - R^2"' )
plt.show()
print('****** best epsilon: ', np.argmin(residuals) + 10)
