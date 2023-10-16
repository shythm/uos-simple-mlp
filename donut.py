# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3
# Experimental Tasks (2) Use donut shaped data

import numpy as np
import layer as lyr
import visualization as vis

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

    # hyperparameters
    lr = 0.2
    activation = 'sigmoid'
    epochs = 20000

    # initialize layers
    hidden_layer = lyr.Layer(2, 4, activation, lr)
    output_layer = lyr.Layer(4, 1, activation, lr)
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

        # save errors to show the error graph
        errors.append(error_sum)

        # save contour
        if epoch % 5000 == 0:
            save_contour(contour_num)
            contour_num += 1

        # test (use the same data)
        print(f'[{epoch + 1}/{epochs}] - error: {error_sum} \r', end="")

    print(f'\nLearning finished. error: {error_sum}')
    save_contour(contour_num)

    for x, y in zip(train_data_x, train_data_y):
        output = lyr.forward_all_layers(x, model)
        print(f'x: {x.T}, y: {y}, y_hat: {output}')

    lyr.save_weights_of_all_layers('logs/donut_weights.txt', model)
    vis.save_error_graph(np.array(errors).reshape(-1), 'logs/donut_error.png')