# 2023-2 Artificial Intelligence Coding

- 2019920037 컴퓨터과학부 이성호

## Assignment 2

- Experiment with learning AND, OR, and XOR gates (two-dimensional input).
- Show the learning process using graphs(x1, x2 two-dimensional straight-line graph).
- Error graph for iterative learning
- Hand in Program code (text), executable, execution environment, result report all in a zip file
- Evaluation: 10% execution, 10% output, 10% comments, 25% completeness, 10% errors, 10% creative, 25% report
- Implement using modules
- Bonus points if implemented as a class. Compose output calculation and learning process as member functions.

## How to run

To run this code, you need to install the following packages: `numpy`, `matplotlib`. You can install them using `pip` with requirements.txt file.
```bash
pip install -r requirements.txt
```

How to run the code:
```bash
python logic_gate.py [gate] [learning rate] [epochs]
Available gates: AND, OR, XOR
This program learns the given gate using single perceptron which has two inputs.
```

Examples:
```bash
python logic_gate.py and 0.1 100
```
```bash
python logic_gate.py or 0.2 100
```
```bash
python logic_gate.py xor 0.5 1000
```
