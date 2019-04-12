import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import math
from random import randint
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
class Smote():
    def __init__(self,distance,range1,range2):
        self.synthetic_arr = []
        self.newindex = 0
        self.distance_measure = distance
        self.range1 =range1
        self.range2 = range2

    def Populate(self, N, i, indices, min_samples, k):
        """
            此代码主要作用是生成增强数组

            Returns:返回增强后的数组
        """

        choice_list =[]
        def choice(data):
            p = []
            wc = {}
            for num in data:
                # print(num)
                wc[num] = wc.setdefault(num, 0) + 1
            for key in wc.keys():
                p.append(wc[key] / len(data))
            # print(p)
            keylist = np.array([key for key in wc.keys()])
            # print(wc[0])
            return keylist,p
        for index in self.range1:
            choice_list.append(choice(min_samples[:,index]))

        while N != 0:
            arr = np.zeros(min_samples[0])
            arr[-2] = min_samples[i][-2]
            arr[-1] = min_samples[i][-1]
            nn = randint(0, k - 2)
            # 统计离散型变量
            for index in self.range1:
                arr[index] = np.random.choice(choice_list[index][0],size=1,p=choice_list[index][1])
            #for attr in features2:
            for attr in self.range2:
                min_samples[i][attr] = float(min_samples[i][attr])
                min_samples[indices[nn]][attr] = float(min_samples[indices[nn]][attr])
                try:
                    diff = float(min_samples[indices[nn]][attr]) - float(min_samples[i][attr])
                except:
                    print('这是第%d列'%attr,min_samples[indices[nn]][attr],min_samples[i][attr])
                gap = random.uniform(0, 1)

                arr[attr] = float(min_samples[i][attr]) + gap * diff
            #print(arr)
            self.synthetic_arr.append(arr)
            self.newindex = self.newindex + 1
            N = N - 1

    def k_neighbors(self, euclid_distance, k):
        nearest_idx_npy = np.empty([euclid_distance.shape[0], euclid_distance.shape[0]], dtype=np.int64)

        for i in range(len(euclid_distance)):
            idx = np.argsort(euclid_distance[i])
            nearest_idx_npy[i] = idx
            idx = 0

        return nearest_idx_npy[:, 1:k]

    def find_k(self, X, k):

        """
               Finds k nearest neighbors using euclidian distance

               Returns: The k nearest neighbor
        """

        euclid_distance = np.empty([X.shape[0], X.shape[0]], dtype=np.float32)

        for i in range(len(X)):
            dist_arr = []
            for j in range(len(X)):
                dist_arr.append(math.sqrt(sum((X[j] - X[i]) ** 2)))
            dist_arr = np.asarray(dist_arr, dtype=np.float32)
            euclid_distance[i] = dist_arr

        return self.k_neighbors(euclid_distance, k)

    def generate_synthetic_points(self, min_samples, N, k):

        """

            Parameters
            ----------
            min_samples : 要增强的数据
            N :要额外生成的负样本的数目
            k : int. Number of nearest neighbours.
            Returns
            -------
            S : Synthetic samples. array,
                shape = [(N/100) * n_minority_samples, n_features].
        """

        if N < 1:
            raise ValueError("Value of N cannot be less than 100%")

        if self.distance_measure not in ('euclidian', 'ball_tree'):
            raise ValueError("Invalid Distance Measure.You can use only Euclidian or ball_tree")

        if k > min_samples.shape[0]:
            raise ValueError("Size of k cannot exceed the number of samples.")

        T = min_samples.shape[0]

        if self.distance_measure == 'euclidian':
            indices = self.find_k(min_samples, k)

        elif self.distance_measure == 'ball_tree':
            nb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(min_samples)
            distance, indices = nb.kneighbors(min_samples)
            indices = indices[:, 1:]

        for i in range(indices.shape[0]):
            self.Populate(N, i, indices[i], min_samples, k)

        return np.asarray(self.synthetic_arr)
