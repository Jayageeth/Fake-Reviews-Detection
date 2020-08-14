from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3d(X, y, clf) :
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    pca = PCA(n_components = 3, whiten = True).fit(X)
    X_pca = pca.transform(X)

    xs_red = []
    ys_red = []
    zs_red = []

    xs_blue = []
    ys_blue = []
    zs_blue = []

    for i in range (0, len(X_pca)) :
        if y[i] == 0 :
            xs_red.append(X_pca[i][0])
            ys_red.append(X_pca[i][1])
            zs_red.append(X_pca[i][2])

        else :
            xs_blue.append(X_pca[i][0])
            ys_blue.append(X_pca[i][1])
            zs_blue.append(X_pca[i][2])
    '''
    print (xs)
    print ("---------------")
    print (ys)
    '''
    ax.scatter(xs_red, ys_red, zs_red, c = 'red', marker = '.', alpha = 0.5)#, s=20, c=None, depthshade=True, *args, **kwargs)
    ax.scatter(xs_blue, ys_blue, zs_blue, c = 'blue', marker = '.', alpha = 0.5)
    plt.show()

def plot2d(X, y, clf):
    pca = PCA(n_components=1, whiten=True).fit(X)
    X_pca = pca.transform(X)

    plt.figure()

    X_axis_red = []
    X_axis_red_points = []

    X_axis_blue = []
    X_axis_blue_points = []

    for i in range (0, 21000) :
        if i % 100 != 0 :
            continue

        if y[i] == 0 :
            X_axis_red.append(i)
            X_axis_red_points.append(X_pca[i])

        else :
            X_axis_blue.append(i)
            X_axis_blue_points.append(X_pca[i])

    plt.scatter(X_axis_red, X_axis_red_points, marker = '.', c = 'coral', alpha = 0.5)
    plt.scatter(X_axis_blue, X_axis_blue_points, marker = '.', c = 'lightblue', alpha = 0.5)

    plt.legend()
    plt.show()

def plot_comp(y_test, y_pred_bernoulli, y_pred_svc) :
        plt.figure()

        X_axis = []
        y_test_plot = []
        y_pred_bernoulli_plot = []
        y_pred_svc_plot = []

        for i in range (0, len(y_test)) :
            if i % 50 != 0 :
                continue

            X_axis.append(i)

            y_test_plot.append(y_test[i])
            y_pred_bernoulli_plot.append(y_pred_bernoulli[i])
            y_pred_svc_plot.append(y_pred_svc[i])

            '''
            if y[i] == 0 :
                X_axis_red.append(i)
                X_axis_red_points.append(X_pca[i])

            else :
                X_axis_blue.append(i)
                X_axis_blue_points.append(X_pca[i])
            '''

        plt.scatter(X_axis, y_test_plot, marker = '.', c = 'blue', alpha = 0.5)
        plt.scatter(X_axis, y_pred_bernoulli_plot, marker = '.', c = 'red', alpha = 0.5)
        plt.scatter(X_axis, y_pred_svc_plot, marker = '.', c = 'green', alpha = 0.5)

        plt.legend()
        plt.show()
