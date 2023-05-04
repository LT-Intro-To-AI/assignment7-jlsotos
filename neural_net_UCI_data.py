from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    out = int(tokens[0])
    output =[]
    if out == 1:
        output.append(0)
    if out == 2:
        output.append(1/5)
    if out== 3:
        output.append(2/5)
    if out==4:
        output.append(3/5)
    if out==5:
        output.append(4/5)

    if out==6:
        output.append(1)
    
    # print(output)
    y=0
    inpt = []
    for x in range(len(tokens[1:])):
        if x!=3:
            tokens[x] = tokens[x].replace("red", '0') 
            tokens[x] = tokens[x].replace("blue", '1')
            tokens[x] = tokens[x].replace("white", "2")
            tokens[x]= tokens[x].replace("gold", "3")
            tokens[x]= tokens[x].replace('green', '4')
            tokens[x]= tokens[x].replace("orange", "5")
            tokens[x] = tokens[x].replace('\n', '')
            tokens[x]= tokens[x].replace("brown", "6")
            tokens[x] = tokens[x].replace("black", '7')
            inpt.append(float(tokens[x]))
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    x=0
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]
            # print(x)
            x+=1
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("flag.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

#for line in training_data:
    #print(line)
td = normalize(training_data)
#print(td)
xtrain, xtest= train_test_split(td)
nn = NeuralNet(27, 3, 1)
nn.train(xtrain, iters=100_000, print_interval=1000, learning_rate=0.1)

for i in nn.test_with_expected(xtest):
    print(f"desired: {i[1]}, actual: {i[2]}")
