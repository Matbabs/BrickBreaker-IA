Matisse BABONNEAU

# BrickBreaker (AI) - Neural Network + Genetic Algorithm

![](https://img.shields.io/static/v1.svg?label=Python&message=Proramming&color=d26667)
![](https://img.shields.io/static/v1.svg?label=Artificial&message=Intelligence&color=cf855b)
![](https://img.shields.io/static/v1.svg?label=AI&message=NeuralNetwork&color=c49058)
![](https://img.shields.io/static/v1.svg?label=AI&message=GeneticAlgorithm&color=b0b451)
![](https://img.shields.io/static/v1.svg?label=Machine&message=Learning&color=68b369)
![](https://img.shields.io/static/v1.svg?label=Deep&message=Learning&color=6066d5)

> The objective of this project is to learn artificial intelligence programming. It is therefore a project of discovery and theoretical approach of the subject. All the solutions presented here are the result of personal choices and do not necessarily correspond to the most efficient choices. 

## Contents

* [Neural Network](#neural-network)
* [Genetic Algorithm](#genetic-algorithm)

# Neural Network

The first phase of the project is the creation of a neural network. Indeed, the final objective is to combine a NN and a genetic algorithm. 

The neural network was therefore developed according to the following model.

![Neural Network](./assets/nn.svg)

| Layer      | Flatten (input) | Dense | Dense | Dense | Dense | Dense | Dense (output) |
|------------|-----------------|-------|-------|-------|-------|-------|----------------|
| Size       | 3               | 128   | 256   | 512   | 256   | 128   | 4              |
| Activation | None            | ReLu  | ReLu  | ReLu  | ReLu  | ReLu  | SoftMax        |


Note that this neural network is a Q-Network. It could be used in a DQN with a Q-learning algorithm. However it is not the method used here. [Tensorflow DQN article.](https://www.tensorflow.org/agents/tutorials/0_intro_rl)

The model can be trained using a simple reinforcement learning strategy, notably by observing the scores produced during training. The model once trained can be able to play. However, the aim is to use a genetic algorithm to improve the weights and biases of the neural network.

# Genetic Algorithm

Neural Networks coupled with Genetic Algorithms can really accelerate the learning process to solve a certain problem. 

One of the most important points of this form of learning is that NN requires a huge amount of data for its learning, whereas GA can perform with less data.

The genetic algorithm thus makes it possible to optimise the performance of a neural network.

The main objective is to implement a life cycle and reproduction generation based on biology: 

> __Selection__:
>
> To determine which individuals are more likely to perform best, a selection is made.
This process is analogous to a natural selection process, with the most adapted individuals winning the reproductive competition while the least adapted die before reproduction, thus improving overall adaptation.
>
>  __Crossover__: 
>
> During this operation, two chromosomes exchange parts of their chains, to give new chromosomes. These crossings can be simple or multiple.
>
> __Mutations__:
>
> In a random way, a gene can mutate within a chromosome.
 
![](./assets/ga.png)

Preparation of the algorithm

* Create a population of several NNs.
* Assign random hyper-parameters (weights and bias) to all the NNs.

Algorithm (for each generation and a candidates population):

1. Play all the NNs simultaneously or one by one.

2. Calculate the performance of each NN based on its cost. Fitness will be used to increase the chances of a NN “reproducing.” 

3. Choose the 2 best NNs. For the 2 next ones you have to crossover genes (weights and bias). 

4. Select some childs to repopulate the next generation.

5. Mutate the genes of the childs. Mutating is required to maintain some amount of randomness in the GA.