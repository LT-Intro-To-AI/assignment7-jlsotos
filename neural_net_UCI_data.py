from typing import Tuple
from neural import *


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    out = int(tokens[0])
    output = [1 if out == 1 else 0.5 if out == 2 else 1 if out== 3 else 1.5 if out==4 else 2 if out==5 else 2.5 if out==6 else 3]
    inpt = [float(i) for x in tokens[1:] for i in x i=0 if i=="red" i=1 if i=="blue" i==2 if i=="white" i=3 if i=="orange" i=4 if i=="black" i=5 if i=="gold" i=6 if i=="brown" i=7 if i=="green"]

    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("flag.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]
for i in range(len(training_data[17][0])):
    training_data[17][0][i]=[1 if training_data[17][0][i]=="green" else 0 if training_data[17][0][i]=="brown" else 1 if training_data[17][0][i]=="red" else 2 if training_data[17][0][i]=="orange" else 3 if training_data[17][0][i]== "blue" else 4]
td = normalize(training_data)

nn = NeuralNet(29, 3, 1)
nn.train(td, iters=100_000, print_interval=1000, learning_rate=0.1)

for i in nn.test_with_expected(td):
    print(f"desired: {i[1]}, actual: {i[2]}")
