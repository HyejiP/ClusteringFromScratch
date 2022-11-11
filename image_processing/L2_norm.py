from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import time

###################### Below are the helper functions ###############################

# this function is for loading an image file and extracting RGB values
def load_file(filename):
    im = io.imread(filename)
    io.imshow(im)
    plt.title('Original Image')
    plt.show()

    # width and height will be used later on to convert pixel values to image file after compressing the pixels
    width, height, num_channel = im.shape
    pixels = im.reshape(width*height, num_channel)
    
    # if the image file consists of four-channel format; R, G, B, and A (instead of RBG), we will remove the last column which contain Alpha values 
    if num_channel == 4:
        pixels = np.delete(pixels, -1, axis=1)
        
    return pixels, width, height


# below function executes K-means algorithm
def find_centroids(pixels, k=2):
    
    # we randomly initialize k cluster centers
    init_cent = pixels[np.random.randint(pixels.shape[0], size=k), :]
    # we will define an empty ndarray with length of pixels, to contanin the cluster assignment results of each pixel
    classes = np.array([None] * len(pixels))

    iter_count = 1
    # below will log the start time to calculate the total elapsed time for convergence
    tic = time.perf_counter()
    
    # we will keep iterating until there is no change of centroids
    while True:
        
        print('--iteration', iter_count)
        # each data point will be assigned to its closest cluster center in terms of l2(Euclidean) distance
        for i in range(len(pixels)):
            dist = np.linalg.norm(pixels[i] - init_cent, axis=1)
            classes[i] = np.argmin(dist)
        
        # each centroid will be moved to the average of all data points that belong to the centroid
        centroid = []
        for n in range(k):
            # if k is too big and there is an empty cluster where no data point is assigned, we will decrement the k value and repeat cluster assignment again
            if len(pixels[classes == n]) == 0:
                return find_centroids(pixels, k-10)
            else:
                centroid.append(pixels[classes == n].mean(axis=0))
        centroid = np.array(centroid)
                    
        # if centroids remain unchanged we will stop iteration. otherwise we will keep updating the centroids 
        if np.all(init_cent == centroid):
            break
        else:
            init_cent = centroid
            
        iter_count += 1
    
    print(f'--Converged after {iter_count} iterations')
    # below will log the end time; so the total elapsed time will be 'toc - tic'
    toc = time.perf_counter()
    elapsed_sec = round(toc - tic, 4)
    
    return classes, centroid, iter_count, elapsed_sec


# this function will return a dictionary with converged centroids; keys are cluster numbers, and values are location of centroids
def centroid_dict(centroid):
    cent_dict = {}
    
    # when we build the dictionary, we will round the location of centroids, as we have to compress the pixels with these later on
    # pixel values have to be integer between 0-255
    for i, c in enumerate(centroid):
        cent_dict[i] = c.astype(int)
        
    return cent_dict


# this function will return compressed pixels referring to class assignment and the dictionary containing location of centroids
def compress_pixels(classes, cent_dict):
    comp_pixels = []
    for i in range(len(classes)):
        comp_pixels.append(cent_dict[classes[i]])
    comp_pixels = np.array(comp_pixels)
    
    return comp_pixels


# this function will show the compressed image
def reproduce_image(comp_pixels):
    comp_image = comp_pixels.reshape(width, height, 3)
    io.imshow(comp_image)
    return plt.show()


###################### Below are the codes to play with different pictures and k values ###############################

im_filenames = ['football.bmp', 'GeorgiaTech.bmp', 'bird.bmp']
k = 2

for filename in im_filenames:
    pixels, width, height = load_file(filename)

    classes, centroid, iter_count, elapsed_sec = find_centroids(pixels=pixels, k=k)
    
    # just in case if the k is too large, we will decrement k value till we have no empty clusters
    # in this case, we might want to calculate the number of clusters to see how many clusters we have
    num_clusters = len(np.unique(classes))
    cent_dict = centroid_dict(centroid)
    comp_pixels = compress_pixels(classes, cent_dict)
    comp_image = reproduce_image(comp_pixels)
    print(f'--filename: {filename} \n--k: {num_clusters} \n--number of iterations: {iter_count} \n--elapsed time: {elapsed_sec} seconds')
