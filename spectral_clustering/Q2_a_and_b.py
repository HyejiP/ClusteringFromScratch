from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ll
import math


###################### Below are helper funcitons ###############################

def load_file(filename):
    im = io.imread(filename)
    im_resized = im[::4, ::4] # sample every fourth pixels
    width, height = im_resized.shape

    pixels = im_resized.reshape(1, width*height) # vectorize pixels 
        
    return pixels, width, height

def centering(matrix):
    mu = np.mean(matrix, axis=0) # mean of each pixels across all the pictures
    centered_matrix = matrix - mu[None, :] # center matrix 
    return centered_matrix, mu.reshape(-1, 1)

# compose a covariance matrix
def eigdecomposition(centered_matrix, k):
    C = np.dot(centered_matrix, centered_matrix.T) / width # covarian matrix C
    eigvals, eigvecs = ll.eigs(C, k=k) # conduct eigendecomposition on matrix C and find the k largest eigenvalues 
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    return eigvals, eigvecs

# plot k eigenfaces with below function
def print_eigfaces(centered_matrix, eigvals, eigvecs, k):
    fig, axs = plt.subplots(1, k, figsize=(5*k, 5))
    for i in range(k):
        dim = np.dot(eigvecs[:, i].T, centered_matrix)/math.sqrt(eigvals[i]) # compute reduced represetation Z_i
        axs[i].imshow(dim.reshape(width, height), cmap='gray_r') # reshape it to 2D 
        axs[i].axis('off')
    return axs


################## Below is code for Q2(a) to print out eigenfaces #####################################


# load each image file and compose a data matrix for each subject
subnames = ['glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'wink']

sub01_files = ['yalefaces/subject01.surprised.gif']
sub02_files = []
for i in subnames:
    filename = 'yalefaces/subject01.' + i + '.gif'
    sub01_files.append(filename)
    filename = 'yalefaces/subject02.' + i + '.gif'
    sub02_files.append(filename)

sub01 = np.empty((0, 4880)) 
sub02 = np.empty((0, 4880))
for filename in sub01_files:
    pixels, width, height = load_file(filename=filename)
    sub01 = np.vstack((sub01, pixels)) # stack each image file's pixels(a row vector of 1*4880) vertically 
for filename in sub02_files:
    pixels, width, height = load_file(filename=filename)
    sub02 = np.vstack((sub02, pixels)) # sub01 is the data matrix of subject01, sub02 is the data matrix of subject02


# center the data matrix using a helper function above
centered_sub01, mu_01 = centering(sub01)
centered_sub02, mu_02 = centering(sub02)

# define K=6, and get the largest 6 eigenvalues and corresponding 6 eigenvectors using a helper function above
K = 6
eigvals_01, eigvecs_01 = eigdecomposition(centered_sub01, k=K)
eigvals_02, eigvecs_02 = eigdecomposition(centered_sub02, k=K)

# print the eigenfaces using a helper function defined above
axs = print_eigfaces(centered_sub01, eigvals_01, eigvecs_01, k=K)
plt.show()
axs = print_eigfaces(centered_sub02, eigvals_02, eigvecs_02, k=K)
plt.show()


################## Below is code for Q2(b) for face recognition #####################################

# load the test images and vectorize pixels
pixels_01, width, height = load_file('yalefaces/subject01-test.gif')
pixels_02, width, height = load_file('yalefaces/subject02-test.gif')

# substract subject mean from vectorized pixels of test images
test_01 = pixels_01.reshape(-1, 1) - mu_01
test_02 = pixels_02.reshape(-1, 1) - mu_02
test_images = [test_01, test_02]

# compute 1st eigenface of each test images
eigface_01 = (np.dot(eigvecs_01[:,0].T, centered_sub01)/math.sqrt(eigvals_01[0])).reshape(-1, 1)
eigface_02 = (np.dot(eigvecs_02[:,0].T, centered_sub02)/math.sqrt(eigvals_02[0])).reshape(-1, 1)
faces = [eigface_01, eigface_02]

S = np.empty(shape=(2, 2))
for j in range(2):
    for i in range(2):
        S[i, j] = (np.linalg.norm(test_images[j] - (faces[i] @ faces[i].T @ test_images[j])))**2
 
print('******* Below is S *******\n', S)
