# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:55:12 2016

@author: zck
"""

import os, cPickle, cv2, math
import numpy as np
import sklearn.cluster

def extract_feature(label_filename, data_root):
    for fpathe, dirs, fs in os.walk(data_root):
        for f in fs:
            name = str(f)
            if name.endswith('.txt'):
                print os.path.join(fpathe, f)

    with open(label_filename,'r') as f:
        for line in f.readlines():
            temp = line.strip().split()
            if( 2==len(temp) ):
                print( temp )
                
                
def generator_centers(data_root, K):
    dense = cv2.FeatureDetector_create("Dense")
    brief = cv2.DescriptorExtractor_create("SIFT")
    
    image_filenames = []
    for fpathe, dirs, fs in os.walk(data_root):
        for f in fs:
            name = str(f)
            if name.endswith('.jpg'):
                image_filenames.append( os.path.join(fpathe, f) )

    cnt = 0;
    points = []
    for image_filename in image_filenames:
        cnt += 1
        if cnt%100 > 0:
            continue
        img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        if img.mean()>250:
            continue
        kp = dense.detect(img, None)
        kp, des = brief.compute(img, kp)
        for t in des:
            points.append(t)
        #print("# kps: {}, descriptors: {}".format(len(kp), des.shape) )
    print( 'sample points: {0}x{1}'.format(len(points), len(points[0]) ) )
    
#    kmeans = sklearn.cluster.KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=3000, tol=0.0001, precompute_distances='auto', n_jobs=-1)    
#    kmeans.fit( np.asarray(points) )
    criteria = (cv2.TERM_CRITERIA_EPS, 3000, 0.0001)
    flags = cv2.KMEANS_PP_CENTERS
    ret, labels, centers = cv2.kmeans( np.asarray(points), 100, criteria, 10, flags)
    print('mse: {0}'.format( math.sqrt(ret)/len(points) ) )
    
    with open('{0}-{1}.sift.feature'.format('train1-100',K), 'w') as f:
        centers.dump(f)
    return centers


if __name__ == '__main__':
    train_label_filename = '..\\..\\CCPR-data\\data\\train\\train_label.txt'
    train_data_root = '..\\..\\CCPR-data\\feature\\train\\'
    
#    extract_feature(label_filename = train_label_filename, data_root = train_data_root)
    centers = generator_centers(data_root = train_data_root, K = 100)
    
#    cv2.KMeans2(points, cluster_count, clusters,
#                   (cv.CV_TERMCRIT_EPS + cv.CV_TERMCRIT_ITER, 10, 1.0))

    
    
    