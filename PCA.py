'''
Created on Jan 30, 2016

@author: amrkoura
'''

# read the data matrix 
from numpy import genfromtxt, sort
import numpy as np
import operator
import copy
import matplotlib.pyplot as plt
from blaze.expr.expressions import Label

#used to find the index of element inside array
def find_index(array, element):
    for index, value in enumerate(array):
        if value==element:
            return index
        
#read the input file
my_data = genfromtxt('data-dimred-X.csv', delimiter=',')
print len(my_data)
print len(my_data[0])
#my_data=my_data.transpose()
#read the labels
my_output=genfromtxt('data-dimred-Y.csv')
# transform to zero mean
mean = my_data.mean(axis=1)
my_data=my_data-  mean[:, np.newaxis]
print len(mean)
# compute covariance matrix
cov_matrix=np.cov(my_data)
# compute eigen value and eigen vector
#eigen value will be vector of 500 values
#eigen vector will contain 500 vector and each vector will be in size 500
eig_values, eig_vectors = np.linalg.eig(cov_matrix)

for i in range(len(eig_values)):
    eigv = eig_vectors[:,i].reshape(1,500).T
    np.testing.assert_array_almost_equal(cov_matrix.dot(eigv),
                                         eig_values[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)

unsorted_eigen_values=copy.copy(eig_values)
eig_values=abs(eig_values)
eig_val_vector_sorted=sorted(eig_values, reverse=True)
print 'U1=',eig_val_vector_sorted[0]
print 'U2=',eig_val_vector_sorted[1]

# get the larget two eigen values u1,u2
index1=find_index(unsorted_eigen_values,eig_val_vector_sorted[0])
index2=find_index(unsorted_eigen_values,eig_val_vector_sorted[1])

# matrix W size is 500 rows and 2 columns
matrix_w= np.hstack((eig_vectors[index1].reshape(500,1),
                    eig_vectors[index2].reshape(500,1))
                    )
# transform the data matrix
transformed_matrix=matrix_w.T.dot(my_data)
print len(transformed_matrix)
print len(transformed_matrix[0])
# plot the data
plt.plot(transformed_matrix[0,0:50], transformed_matrix[1,0:50],
         'o', markersize=7, color='blue', alpha=0.5,Label='class 1')

plt.plot(transformed_matrix[0,50:100], transformed_matrix[1,50:100],
         '^', markersize=7, color='red', alpha=0.5,Label='class 2')

plt.plot(transformed_matrix[0,100:150], transformed_matrix[1,100:150],
         'v', markersize=7, color='black', alpha=0.5,Label='class 3')




plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed data into 2D')

plt.show()

print "finish"

  





