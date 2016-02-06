#solution of the second part of task 3.3 on dimension reduction using LDA reduction algorothm

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
import timeit


# compute the between-class scatter matrix
def compute_between_scatter_matrix(data, labels, num_classes):
    data_dim = data[:,0].size
    scatter_matrix = np.zeros((data_dim, data_dim))
    means_per_class  = np.zeros((num_classes, data_dim))# the mean of each dimension for label k
    diff_matrix = np.zeros((num_classes, data_dim))
    for k in range(1,num_classes+1):
        n_k = len(labels[labels==k]) # number of instances belonging to class k
        for i in range(data_dim):
            mean_per_class = np.mean(data[i,labels==k]) # the mean of each dimension for label k
            total_mean = np.mean(data[i, :]) # total mean, regardless of class
            means_per_class[k-1, i] = mean_per_class
            diff_matrix[k-1,i] = mean_per_class - total_mean
    for k in range(1,num_classes+1):
        scatter_matrix = scatter_matrix + n_k * diff_matrix[k-1,:].T * diff_matrix[k-1,:] 
    print 's', scatter_matrix.shape
    return scatter_matrix
    
    # compute the in-class scatter matrix
def compute_in_scatter_matrix(data, labels, num_classes):
    data_dim = data[:, 0].size
    scatter_matrix = np.zeros((data_dim, data_dim))
    for k in range(1, num_classes+1):
        cov_per_class = np.cov(data[:, labels==k]) # the covariance matrix for each class label k
        scatter_matrix = scatter_matrix + cov_per_class
    return scatter_matrix

   

if __name__ == "__main__":

    data = np.loadtxt('data-dimred-X.csv',dtype=np.float,comments='#',delimiter=', ')
    labels = np.loadtxt('data-dimred-y.csv',dtype=np.float,comments='#',delimiter=', ')
    #print data[:,0].size #500
    #print data[0,:].size #150 instances

    #performing dimensionality reduction using LDA
    

    #Compute the within class scatter matrix SW
    k = 3 # label space size
    #S_W = compute_in_scatter_matrix(data, labels, k)
    # computng S_W
    
    mean_vectors = []
    for k in range(1, 4):
        mean_vectors.append(np.mean(data[:, labels==k], axis=1))
        
        
    
    S_W = np.zeros((500,500))
    for cl,mv in zip(range(1,4), mean_vectors):
        class_sc_mat = np.zeros((500,500))                  # scatter matrix for every class
        for row in data[:, labels==cl].T:
            row, mv = row.reshape(500,1), mv.reshape(500,1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += (1/50.0)*class_sc_mat              
    
    
    
    #S_B =   (data, labels, k)
    
    
    
    
    
    
    #data=data.T
    overall_mean = np.mean(data, axis=1)
    
    S_B = np.zeros((500,500))
    for i,mean_vec in enumerate(mean_vectors):
        n = data[:, labels==(i+1)].shape[0]
        mean_vec = mean_vec.reshape(500,1) # make column vector
        overall_mean = overall_mean.reshape(500,1) # make column vector
        S_B +=  (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    
    
    #S_W_B = np.dot(np.linalg.inv(S_W), S_B) #SW^-1 SB
    S_W_B = np.linalg.inv(S_W).dot(S_B) #SW^-1 SB
    eigen_vals, eigen_vec = np.linalg.eigh(S_W_B)
    #eigen_vals, eigen_vec = np.linalg.eigh(S_W_B)
    #eigen_vals, eigen_vec = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    
    # 2 eigen vectors of the largest eigen values
    #u1 = eigen_vec[eigen_vec.shape[0]-1]
    #u2 = eigen_vec[eigen_vec.shape[0]-2]
    
    u1 = eigen_vec[:, eigen_vec.shape[0]-1]
    #u2 = eigen_vec[:, eigen_vec.shape[0]-1]
    u2 = eigen_vec[:,eigen_vec.shape[0]-2]
    
    U_2 = np.vstack((u1, u2))
       # projecting the normalized data into R^2
    reduced_dim_data_lda = np.dot(U_2, data)
    #print reduced_dim_data_lda.shape

    plt.scatter(reduced_dim_data_lda[0,labels==1], reduced_dim_data_lda[1, labels==1], color= 'b',  marker='*', s=50, label='class 1')
    plt.scatter(reduced_dim_data_lda[0,labels==2], reduced_dim_data_lda[1, labels==2], color= 'r',  marker='*', s=50, label='class 2')
    plt.scatter(reduced_dim_data_lda[0,labels==3], reduced_dim_data_lda[1, labels==3], color= 'g',  marker='*', s=50, label='class 3')
    plt.legend(loc='upper left')
    plt.title('reduced dim data in R^2 using LDA')   
    plt.show()

    # now project to R^3
    u3 = eigen_vec[:,eigen_vec.shape[0]-3]
    U_3 = np.vstack((u1, u2, u3))
    reduced_dim_data_3 = np.dot(U_3, data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #print 'shape', reduced_dim_data_3.shape
    ax.scatter(reduced_dim_data_3[0,labels==1],reduced_dim_data_3[1,labels==1],reduced_dim_data_3[2,labels==1],c="b",marker='o', s=50, label = 'class 1')
    ax.scatter(reduced_dim_data_3[0,labels==2],reduced_dim_data_3[1,labels==2],reduced_dim_data_3[2,labels==2],c="r",marker='o', s=50, label = 'class 1')
    ax.scatter(reduced_dim_data_3[0,labels==3],reduced_dim_data_3[1,labels==3],reduced_dim_data_3[2,labels==3],c="g",marker='o', s=50, label = 'class 1')
    plt.legend(loc='upper left')
    plt.title('reduced dim data in R^3 using LDA')
    plt.show()
    print 'finish'