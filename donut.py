# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3
# Experimental Tasks (2) Use donut shaped data

import sys
import numpy as np
import layer as lyr
import visualization as vis
from tqdm import tqdm

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

if __name__ == '__main__':
    # validation of arguments
    if len(sys.argv) != 6:
        print(f'Usage: python {sys.argv[0]} '
               '[hidden nodes] [learning rate] [activation] [epochs] [check epoch]')
        print('Example: python donut.py 4 0.1 sigmoid 10000 2000')
        print('Available activation functions: sigmoid, relu, step')
        print('This program learns the given donut shaped data using two-layer perceptron.')
        sys.exit(-1)

    # hyperparameters and variables
    hidden_nodes = int(sys.argv[1]) # ex: 4
    lr = float(sys.argv[2]) # ex: 0.1
    activation = sys.argv[3] # ex: sigmoid
    epochs = int(sys.argv[4]) # ex: 10000
    check_epoch = int(sys.argv[5]) # ex: 2000

    # initialize layers
    hidden_layer = lyr.Layer(2, hidden_nodes, activation, lr)
    output_layer = lyr.Layer(hidden_nodes, 1, activation, lr)
    model = [hidden_layer, output_layer]

    # train data and errors
    train_data_x = [np.array(x).reshape(2, 1) for x in donut_shaped_x]
    train_data_y = donut_shaped_y
    errors = []

    # meshgrid to draw contour
    meshgrid_size = (100, 100)
    x = np.linspace(-0.5, 1.5, meshgrid_size[0])
    y = np.linspace(-0.5, 1.5, meshgrid_size[1])
    meshgrid = np.meshgrid(x, y)
    contour_num = 1
    def save_contour(num):
        vis.save_contour_plot_nto1(
            model, f"logs/donut_contour_{num}.png", meshgrid,
            [np.array(donut_shaped_x), np.array(donut_shaped_y)]
        )

    # progressbar
    tqdm.write(f'Learning {len(train_data_x)} samples with {epochs} epochs...')
    
    # learning
    for epoch in (pbar := tqdm(range(epochs))):
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

        # save errors to show the error graph
        errors.append(error_sum)

        # save contour
        if epoch % 5000 == 0:
            save_contour(contour_num)
            contour_num += 1

        # print error
        pbar.set_description(f'Error={error_sum[0, 0]:.6f}')

    print(f'Learning finished. error: {error_sum}')
    save_contour(contour_num)

    for x, y in zip(train_data_x, train_data_y):
        output = lyr.forward_all_layers(x, model)
        print(f'x: {x.T}, y: {y}, y_hat: {output}')

    lyr.save_weights_of_all_layers('logs/donut_weights.txt', model)
    vis.save_error_graph(np.array(errors).reshape(-1), 'logs/donut_error.png')