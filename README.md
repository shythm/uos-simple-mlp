# 2023-2 Artificial Intelligence Coding

- 2019920037 컴퓨터과학부 이성호

## Assignment 3

- Experimental Tasks

  1. Implement AND, OR, and XOR gates
  2. Use donut shaped data (next slide)

- Let the number of layers and the number of nodes per layer be variables.
- Implement the calculation of one layer as a function.
- Save the weight to a file in a matrix format
- Show the learning process (X1, X2 two-dimensional straight line graphs).
  - Plot straight lines of a few nodes.
  - In this repository, it shows the learning process using **contour plot** (instead of straight line graph).
- Show error graph as a function of iteration

## How to run

To run this code, you need to install the following packages: `numpy`, `matplotlib`. You can install them using `pip` with requirements.txt file.

```bash
pip install -r requirements.txt
```

How to run the code:

```bash
$ python main.py
Usage: python main.py [dataset] [hidden nodes] [activation] [learning rate] [epochs] [check epoch]
Example: python main.py donut 4 sigmoid 0.1 10000 2000
Available activation functions: sigmoid, relu, step
Available dataset: donut, and, or, xor
This program learns the given data using two-layer perceptron.
```

Examples:

```bash
python main.py donut 4 sigmoid 0.4 10000 2000
```

```bash
python main.py xor 4 sigmoid 0.4 10000 2000
```
