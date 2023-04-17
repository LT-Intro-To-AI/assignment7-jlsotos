from neural import NeuralNet

print("\n\nTraining SQ\n\n")
sq_training_data= [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
    ([.9], [.81])
]
sqn= NeuralNet(1,20,1)
sqn.train(sq_training_data)

print()
print(sqn.test_with_expected(sq_training_data))
print(sqn.evaluate([0.66]))
print(sqn.evaluate([0.95]))

print()
print("\n\nTraining xor\n\n")
print()

x_or_trainingdata= [
    ([0, 0], [0]),
   ([1,0], [1]),
    ([0,1], [1]),
    ([1,1], [0])
]
xorn= NeuralNet(2,20,1)

xorn.train(x_or_trainingdata)

print()

print(xorn.test_with_expected(x_or_trainingdata))

print("\n\nTraining voter opinoion\n\n")

voter_opinion_data= [
    ([.9, .6, .8, .3, .1], [1]),
    ([.8, .8, .4, .6, .4], [1]),
    ([.7, .2, .4, .6, .3], [1]),
    ([.5, .5, .8, .4, .8], [0]),
    ([.3, .1, .6, .8, .8], [0]),
    ([.6, .3, .4, .3, .6], [0])

]
von= NeuralNet(5,6,1)
von.train(voter_opinion_data)
test_data= [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6]
]
for i in range(len(test_data)):
    print(f"{test_data[i]} evlauates to: {von.evaluate(test_data[i])}")