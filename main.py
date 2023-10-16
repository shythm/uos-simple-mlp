# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3
# Experimental Tasks (1) Implement AND, OR, and XOR gates
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

# gate data
gate_x = [
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
]
and_gate_y = [0, 0, 0, 1] # and gate output
or_gate_y = [0, 1, 1, 1] # or gate output
xor_gate_y = [0, 1, 1, 0] # xor gate output

if __name__ == '__main__':
    # validation of arguments
    if len(sys.argv) != 7:
        print(f'Usage: python {sys.argv[0]} '
               '[dataset] [hidden nodes] [activation] [learning rate] [epochs] [check epoch]')
        print(f'Example: python {sys.argv[0]} donut 4 sigmoid 0.1 10000 2000')
        print('Available activation functions: sigmoid, relu, step')
        print('Available dataset: donut, and, or, xor')
        print('This program learns the given data using two-layer perceptron.')
        sys.exit(-1)

    # hyperparameters and variables
    dataset = sys.argv[1] # what data to learn(ex: donut)
    hidden_nodes = int(sys.argv[2]) # number of hidden nodes(ex: 4)
    activation = sys.argv[3] # activation function(ex: sigmoid)
    lr = float(sys.argv[4]) # learning rate(ex: 0.1)
    epochs = int(sys.argv[5]) # number of epochs(ex: 10000)
    check_epoch = int(sys.argv[6]) # plot model every check_epoch epochs(ex: 2000)

    # initialize layers
    hidden_layer = lyr.Layer(2, hidden_nodes, activation, lr)
    output_layer = lyr.Layer(hidden_nodes, 1, activation, lr)
    model = [hidden_layer, output_layer]

    # prepare training sample data
    train_data_x = gate_x
    if dataset.lower() == 'and':
        train_data_y = and_gate_y
    elif dataset.lower() == 'or':
        train_data_y = or_gate_y
    elif dataset.lower() == 'xor':
        train_data_y = xor_gate_y
    elif dataset.lower() == 'donut':
        train_data_x = donut_shaped_x
        train_data_y = donut_shaped_y
    else:
        print('Unknown dataset:', dataset)
        sys.exit(-1)

    # initialize error list
    errors = []

    # meshgrid to draw contour
    meshgrid_size = (100, 100)
    x = np.linspace(-0.5, 1.5, meshgrid_size[0])
    y = np.linspace(-0.5, 1.5, meshgrid_size[1])
    meshgrid = np.meshgrid(x, y)
    contour_num = 1
    def save_contour(num):
        vis.save_contour_plot_nto1(
            model, f"logs/{dataset}_contour_{num}.png", meshgrid,
            [np.array(train_data_x), np.array(train_data_y)]
        )

    # progressbar
    tqdm.write(f'Learning {len(train_data_x)} samples with {epochs} epochs...')
    
    # a list of input vectors which is ndarray
    train_input_vectors = [np.array(x).reshape(2, 1) for x in train_data_x]

    # learning
    for epoch in (pbar := tqdm(range(epochs))):
        error_sum = 0

        # train
        for x, y in zip(train_input_vectors, train_data_y):
            # forward
            output = lyr.forward_all_layers(x, model)
            # backward
            # error function Error(y_hat) : 1 / 2 * (y - y_hat)^2
            # derivative of error function : (y - y_hat) * (-1)
            cost_error = output - y
            lyr.backward_all_layers(cost_error, model)
            cost = (-cost_error) ** 2 / 2
            error_sum += cost

        # save errors to show the error graph
        errors.append(error_sum)

        # save contour
        if epoch % check_epoch == 0:
            save_contour(contour_num)
            contour_num += 1

        # print error
        pbar.set_description(f'Error={error_sum[0, 0]:.6f}')

    print(f'Learning finished. error: {error_sum}')
    save_contour(contour_num)

    for x, y in zip(train_input_vectors, train_data_y):
        output = lyr.forward_all_layers(x, model)
        print(f'x: {x.T}, y: {y}, y_hat: {output}')

    lyr.save_weights_of_all_layers(f'logs/{dataset}_weights.txt', model)
    vis.save_error_graph(np.array(errors).reshape(-1), f'logs/{dataset}_error.png')