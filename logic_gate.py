# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #2

# Experiment with learning AND, OR and XOR gates (two-dimensional input).
# Show the learning process using graphs (x1, x2 two-dimensional straight-line graph).
# Error graph for iterative learning

import sys
import numpy as np
import matplotlib.pyplot as plt
import perceptron as pcn

# show the learning process using graphs (x1, x2 two-dimensional straight-line graph)
# draw a straight line (w1 * x1 + w2 * x2 + w0 = 0) from weights of each epoch
# and plot four x points[(0, 0), (0, 1), (1, 0), (1, 1)] on the graph
def plot_learning_process(logger, x, y):
    # create a graph
    graph = plt.figure(figsize=(6, 5)).add_subplot(1, 1, 1)

    # plot four x points
    graph.scatter(x[:, 1], x[:, 2], c=y)

    # plot consecutive straight line
    for i in range(len(logger)):
        # get weights of each epoch
        w0, w1, w2 = logger[i][1]
        # draw x2 = -(w1 * x1 + w0) / w2 graph which color is gradually increased red and dotted
        x1 = np.linspace(-0.5, 1.5, 10)
        x2 = -(w1 * x1 + w0) / w2
        linestyle = 'solid' if i == len(logger) - 1 else 'dotted' # last line is solid
        graph.plot(x1, x2, color=(1, 0, 0, i / len(logger)), linestyle=linestyle)

    # limit x, y axis
    graph.set_xlim(-0.5, 1.5)
    graph.set_ylim(-0.5, 1.5)

    # save the graph
    plt.savefig('plot_learning_process.png')

# show error graph for iterative learning
def plot_error_graph(logger):
    # create a new graph
    graph = plt.figure(figsize=(6, 5)).add_subplot(1, 1, 1)
    # plot error
    graph.plot([e for e, _ in logger])
    # add labels
    graph.set_xlabel('epoch')
    graph.set_ylabel('error')
    # save the graph
    plt.savefig('plot_error_graph.png')

if __name__ == '__main__':
    # get arguments from shell
    if len(sys.argv) != 4:
        print('Usage: python logic_gate.py [gate] [learning rate] [epochs]')
        print('Available gates: AND, OR, XOR')
        print('This program learns the given gate using single perceptron which has two inputs.')
        sys.exit(-1)

    # first argument: what gate to learn
    gate = sys.argv[1]
    # second argument: learning rate
    lr = float(sys.argv[2])
    # third argument: number of epochs
    epochs = int(sys.argv[3])

    # prepare training samples
    if gate.lower() == 'and':
        x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        y = np.array([0, 0, 0, 1])
    elif gate.lower() == 'or':
        x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        y = np.array([0, 1, 1, 1])
    elif gate.lower() == 'xor':
        x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        y = np.array([0, 1, 1, 0])
    else:
        print('Unknown gate:', gate)
        sys.exit(-1)

    # create perceptron
    # three dimensional input (bias, x1, x2)
    # initialize weights with small random values (-1 ~ 1)
    # activation function: step function (0 if x < 0, 1 if x >= 0)
    # differential function: step function differential (let's assume it is 1)
    p = pcn.Perceptron(3, 1 - 2 * np.random.rand(3), pcn.step, pcn.step_diff)

    # logger: the list of tuple (error, weights) for plotting error graph
    logger = []

    # print initial weights
    print(f'Learning started({gate.upper()} gate).')
    print('Initial weights:', p.weight)

    # repeat until error is zero or epoch is over
    for epoch in range(epochs):
        # for each training data(x_i, y)
        for i in range(len(x)):
            # input vector
            p.invec = x[i]
            # calculate output - forward propagation
            p.forward()
            # backward propagation - calculate local gradient
            p.backward()
            # update weights(for each weight, adjust it by lr * (target - output) * df/dw)
            p.weight += lr * (y[i] - p.out) * p.localgrad_weight
        
        # calculate error
        error = 0
        for _x, _y in zip(x, y):
            if _y != p.forward(_x):
                error += 1

        # print epoch, error, weights
        print(f'[{epoch + 1}/{epochs}] - error: {error}, weights: {p.weight}')

        # log error and weights
        logger.append((error, p.weight.copy()))

        # if error is zero, stop learning
        if error == 0:
            break

    # try each input and print output
    print('Learning finished.')
    print('Try each input ...')
    for _x in x:
        print(f'{_x} -> {p.forward(_x)}')

    # plot learning process
    plot_learning_process(logger, x, y)
    # plot error graph
    plot_error_graph(logger)
