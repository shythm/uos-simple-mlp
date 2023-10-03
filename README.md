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