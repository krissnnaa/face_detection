import os
import time
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from skcuda import linalg
# Sequential Computation
def sequential_calculation(train_data):
    # train_data = train_data - train_data.mean(axis=1, keepdims=True)
    # cov_mat=np.cov(train_data)
    # np.save('cov_mat.npy', cov_mat)
    # va_eig,ve_eig=lg.eig(cov_mat)
    # val_eig= np.asarray(va_eig)
    # vec_eig=np.asarray(ve_eig)
    # np.save('eig_val.npy',val_eig)
    # np.save('eig_vec.npy',vec_eig)
    cov_mat=np.load('cov_mat.npy')
    val_eig=np.load('eig_val.npy')
    vec_eig=np.load('eig_vec.npy')
    idx = val_eig.argsort()[::-1]
    aval_eig = val_eig[idx]
    avec_eig = vec_eig[:, idx].T
    pca_vec=[]
    total=sum(val_eig)
    eig_sum=0
    for i in range(0,len(aval_eig)):
        eig_sum = eig_sum+aval_eig[i]
        if (eig_sum/total)<0.9:
            pca_vec.append(avec_eig[i])
        else:
            break


    # train_data_pca=np.multiply(np.transpose(vec_eig),train_data)
    # test_data_pca = np.multiply(np.transpose(vec_eig), test_data)

    # Display first eigenface
    pca_vec = np.asarray(pca_vec)
    pca_vec = np.real(pca_vec)
    dis_eig=np.real(avec_eig[0])
    dis_eig[dis_eig<0]=0
    # amax=np.amax(dis_eig)
    # factor= 255/(amax)
    # dis_vec=np.multiply(factor,dis_eig)
    # mat_array = np.asarray(np.uint8(np.reshape(dis_vec, (112,92))))
    # dis_eig = cv2.normalize(dis_eig, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # cv2.imshow("EIGEN FACE",np.reshape(dis_eig,(112,92)))

    # Compute Ecludean Distance

# GPU Computation
def gpu_calculation(train_data):
    train_data = train_data - train_data.mean(axis=1, keepdims=True)
    cov_mat=np.cov(train_data)
    np.save('gpu_cov_mat.npy', cov_mat)
    cov_mat=gpuarray.to_gpu(cov_mat)
    va_eig, ve_eig =linalg.eig(cov_mat, 'V', 'N')
    val_eig= np.asarray(va_eig)
    vec_eig=np.asarray(ve_eig)
    np.save('gpu_eig_val.npy',val_eig)
    np.save('gpu_eig_vec.npy',vec_eig)
    # cov_mat=np.load('gpu_cov_mat.npy')
    # val_eig=np.load('gpu_eig_val.npy')
    # vec_eig=np.load('gpu_eig_vec.npy')
    idx = val_eig.argsort()[::-1]
    aval_eig = val_eig[idx]
    avec_eig = vec_eig[:, idx].T
    pca_vec=[]
    total=sum(val_eig)
    eig_sum=0
    for i in range(0,len(aval_eig)):
        eig_sum = eig_sum+aval_eig[i]
        if (eig_sum/total)<0.9:
            pca_vec.append(avec_eig[i])
        else:
            break


    # train_data_pca=np.multiply(np.transpose(vec_eig),train_data)
    # test_data_pca = np.multiply(np.transpose(vec_eig), test_data)

    # Display first eigenface
    pca_vec = np.asarray(pca_vec)
    pca_vec = np.real(pca_vec)
    dis_eig=np.real(avec_eig[0])
    dis_eig[dis_eig<0]=0
    # amax=np.amax(dis_eig)
    # factor= 255/(amax)
    # dis_vec=np.multiply(factor,dis_eig)
    # mat_array = np.asarray(np.uint8(np.reshape(dis_vec, (112,92))))
    #dis_eig = cv2.normalize(dis_eig, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #cv2.imshow("EIGEN FACE",np.reshape(dis_eig,(112,92)))

    # Compute Ecludean Distance


if __name__ == '__main__':

    select=2
    #Read Images for training and testing
    train_data=np.load('train_data.npy')
    test_data=np.load('test_data.npy')

    if (select == 1):
        start_time = time.time()
        sequential_calculation(train_data)
        elapsed_time = time.time() - start_time

    if (select == 2):
        start_time = time.time()
        gpu_calculation(train_data)
        elapsed_time = time.time() - start_time

print "Total Time Consumed=",elapsed_time
#cv2.waitKey(0)

