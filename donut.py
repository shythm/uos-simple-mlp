# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3
# Experimental Tasks (2) Use donut shaped data

## Requirements ##
# Save the weight to a file in a matrix format
# Show the learning process (X1, X2 two-dimensional straight line graphs).
# Plot straight lines of a few nodes.
# Show error graph as a function of iteration

import sys
import numpy as np
import matplotlib.pyplot as plt
import layer as lyr

# donut shaped data
donut_shaped_x = [
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
    [0.5, 1.],
    [1., 0.5],
    [0., 0.5],
    [0.5, 0.],
    [0.5, 0.5],
]
donut_shaped_y = [
    0, 0, 0, 0, 0, 0, 0, 0, 1
]

# show the learning process using contour plot (instead of straight line graph)
def plot_contour(model: list[lyr.Layer]):
    # create a graph
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor('white')
    ax = fig.add_subplot()

    # create x, y axis
    size = 100
    x = np.linspace(-0.5, 1.5, size)
    y = np.linspace(-0.5, 1.5, size)
    X, Y = np.meshgrid(x, y)
    
    # create z axis
    Z = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            output = lyr.forward_all_layers(np.array([X[i, j], Y[i, j]]).reshape(2, 1), model)
            Z[i, j] = output[0, 0]

    # draw contour plot
    contour1 = ax.contour(X, Y, Z, levels=10, colors='k', linewidths=1, linestyles='--') ## 등고선
    ax.clabel(contour1, contour1.levels, inline=True) ## contour 라벨
    
    # contour2 = ax.contourf(X, Y, Z, levels=256, cmap='jet')
    # fig.colorbar(contour2, shrink=0.5) ## 컬러바 크기 축소 shrink

    # print donut shaped data
    points = np.array(donut_shaped_x)
    ax.scatter(points[:, 0], points[:, 1], c=donut_shaped_y)
    
    plt.savefig('plot_contour.png')

if __name__ == '__main__':

    # hyperparameters
    lr = 0.2
    activation = 'sigmoid'
    epochs = 10000

    # initialize layers
    hidden_layer = lyr.Layer(2, 4, activation, lr)
    output_layer = lyr.Layer(4, 1, activation, lr)
    model = [hidden_layer, output_layer]

    train_data_x = [np.array(x).reshape(2, 1) for x in donut_shaped_x]
    train_data_y = donut_shaped_y

    for epoch in range(epochs):
        error_sum = 0

        # train
        for x, y in zip(train_data_x, train_data_y):
            # < forward >
            output = lyr.forward_all_layers(x, model)

            # < backward >
            # error function Error(y_hat) : 1 / 2 * (y - y_hat)^2
            # derivative of error function : (y - y_hat) * (-1)
            cost_error = output - y
            lyr.backward_all_layers(cost_error, model)

            cost = (-cost_error) ** 2 / 2
            error_sum += cost

        # test (use the same data)
        print(f'[{epoch + 1}/{epochs}] - error: {error_sum} \r', end="")

    print(f'\nLearning finished. error: {error_sum}')

    for x, y in zip(train_data_x, train_data_y):
        output = lyr.forward_all_layers(x, model)
        print(f'x: {x.T}, y: {y}, y_hat: {output}')

    plot_contour(model)