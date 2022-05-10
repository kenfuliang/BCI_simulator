# BCI Simulator

BCI simulator is a platform to simulator brain-machine interface decoder performance. This platform is developed upon Stable Baselines. 

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).

This platform will make it easier for the research community and industry to develop and evaluate BCI decoders, and will serve as a benchmark to have cross comparison. We expect this platform will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of this platform will allow more researcher to experiment with a more advanced idea, without being buried in monkey experiments. 

We performed four representative decoder algorithms. 
[Placeholder: cite]

## Installation
<!-- **Note:** Stabe-Baselines supports Tensorflow versions from 1.8.0 to 1.14.0. Support for Tensorflow 2 API is planned.

### Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows -->

#### Ubuntu

```bash

sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

### Install using anaconda and pip
Install the Stable Baselines package:
```
pip install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


## Example


