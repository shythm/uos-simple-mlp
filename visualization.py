# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3

## Requirements ##
# (1) Show the learning process (X1, X2 two-dimensional straight line graphs).
#     - Plot straight lines of a few nodes.
#     -> This code shows the learning process using contour plot (instead of straight line graph). 
# (2) Show error graph as a function of iteration

import numpy as np
import matplotlib.pyplot as plt
import layer as lyr

# show contour plot of a model
def save_contour_plot_nto1(
        model: list[lyr.Layer],
        filename: str,
        meshgrid: list[np.ndarray],
        points: list[np.ndarray]):
    
    # create a graph
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor('white')
    ax = fig.add_subplot()

    # create x, y axis
    X, Y = meshgrid
    
    # create z axis
    Z = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            output = lyr.forward_all_layers(np.array([X[i, j], Y[i, j]]).reshape(2, 1), model)
            Z[i, j] = output[0, 0]

    # draw contour plot
    contour1 = ax.contour(X, Y, Z, levels=10, colors='k', linewidths=1, linestyles='--') # contour line
    ax.clabel(contour1, contour1.levels, inline=True) # contour line label

    # print and save donut shaped data
    ax.scatter(points[0][:, 0], points[0][:, 1], c=points[1])
    plt.savefig(filename)

# show error graph as a function of iteration
def save_error_graph(
        errors: list[float],
        filename: str):
    
    # create a new graph
    fig = plt.figure(figsize=(6, 5)).add_subplot(1, 1, 1)
    # plot error
    fig.plot(np.arange(1, len(errors) + 1), errors)
    # add labels
    fig.set_xlabel('epoch')
    fig.set_ylabel('error')
    # save the graph
    plt.savefig(filename)
