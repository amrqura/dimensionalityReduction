# solution of task 3.3 on data dim reduction using PCA
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, PathPatch
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d
import numpy.linalg as la
import math
import itertools
from scipy.cluster.vq import kmeans, kmeans2, vq, whiten
import  matplotlib


def normalize_data(data):
    data_dim = data[:,0].size
    means = np.zeros(data_dim)
    normalized_data = np.zeros(data.shape)
    for i in range(data_dim):
        mean = np.mean(data[i,:]) # the mean of each dimension
        means[i] = mean
        normalized_data[i,:] = data[i,:] - mean
    return normalized_data


if __name__ == "__main__":
    matplotlib.matplotlib_fname()
    data = np.loadtxt('data-dimred-X.csv',dtype=np.float,comments='#',delimiter=', ')
    labels = np.loadtxt('data-dimred-y.csv',dtype=np.float,comments='#',delimiter=', ')
    #print data[:,0].size #500
    #print data[0,:].size #150

    #performing dimensionality reduction using PCA
    
    #normalize the data in X to zero
    normalized_data = normalize_data(data)
    #mean = data.mean(axis=1)
    #normalized_data=data-  mean[:, np.newaxis]
    
    
    
    num_data_points = data[0,:].size
    total=np.dot(normalized_data, (normalized_data.T))
    print total[0]
    cov_matrix =  (total)
    print 'now' 
    print cov_matrix[0] 
    print cov_matrix.shape
    eigen_vals, eigen_vec = np.linalg.eigh(cov_matrix)
    
    
   
    
    #print(eig_pairs[0][0])
    #print(eig_pairs[1][0])
    
    # 2 eigen vectors of the largest eigen values
    #u1 = eigen_vec[eigen_vec.shape[0]-1]
    u1 = eigen_vec[:, eigen_vec.shape[0]-1]
    #u2 = eigen_vec[:, eigen_vec.shape[0]-1]
    u2 = eigen_vec[:,eigen_vec.shape[0]-2]
    
    #print(eigen_vals[eigen_vec.shape[0]-1])
    #print(eigen_vals[eigen_vec.shape[0]-2])
    
    U_2 = np.vstack((u1, u2))
    #U_2 = np.vstack((u2, u1))
    #print sum(np.multiply(u1, u2))
    #print U_2.shape
    # projecting the normalized data into R^2
    reduced_dim_data = np.dot(U_2, normalized_data)
    #print reduced_dim_data.shape
    plt.scatter(reduced_dim_data[0,0:50], reduced_dim_data[1, 0:50], color= 'b',  marker='*', s=50, label='class 1')
    plt.scatter(reduced_dim_data[0, 50:100], reduced_dim_data[1, 50:100], color= 'r',  marker='*', s=50, label='class 2')
    plt.scatter(reduced_dim_data[0, 100:150], reduced_dim_data[1, 100:150], color= 'g',  marker='*', s=50, label='class 3')
    plt.legend(loc='upper left')
    plt.title('reduced dim data in R^2 using PCA')   
    plt.show()


    # now project to R^3
    u3 = eigen_vec[eigen_vec.shape[0]-3]
    U_3 = np.vstack((u1, u2, u3))
    reduced_dim_data_3 = np.dot(U_3, normalized_data)
    print reduced_dim_data_3.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print 'shape', reduced_dim_data_3.shape
    ax.scatter(reduced_dim_data_3[0,labels==1],reduced_dim_data_3[1,labels==1],reduced_dim_data_3[2,labels==1],c="b",marker='o', s=50, label = 'class 1')
    ax.scatter(reduced_dim_data_3[0,labels==2],reduced_dim_data_3[1,labels==2],reduced_dim_data_3[2,labels==2],c="r",marker='o', s=50, label = 'class 1')
    ax.scatter(reduced_dim_data_3[0,labels==3],reduced_dim_data_3[1,labels==3],reduced_dim_data_3[2,labels==3],c="g",marker='o', s=50, label = 'class 1')
    plt.legend(loc='upper left')
    plt.title('reduced dim data in R^3 using PCA')
    plt.show()
    print 'finish'