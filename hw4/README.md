# Machine Learning Homework 4

## Description

### PCA

PCA演算法步驟：

1. 標準化d維原"數據集"
2. 建立共變異數矩陣(covariance matrix)
3. 分解共變異數矩陣(covariance matrix)為特徵向量(eigenvector)與特徵值(eigenvalues)。
4. 選取k個最大特徵值(eigenvalues)相對應k個的特徵向量(eigenvector)，其中k即為新特徵子空間的維數。
5. 使用排序最上面的k個的特徵向量(eigenvector)，建立投影矩陣(project matrix)W。
6. 使用投影矩陣(project matrix)W轉換原本d維的原數據至新的k維特徵子空間。

### Image Clustering

1. PCA降維
2. Use T-SNE to compute similarity between all pairs of x
3. K-means to cluster data

## Reference

PCA

[機器學習(6)--主成分分析(Principal component analysis，PCA)](http://arbu00.blogspot.tw/2017/02/6-principal-component-analysispca.html)
[Two-Dimensional PCA:
A New Approach to Appearance-Based
Face Representation and Recognition](http://ira.lib.polyu.edu.hk/bitstream/10397/190/1/137.pdf)

Image Clustering

[sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
[sklearn.cluster.KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
