# BCI Simulator

BCI simulator is a platform to simulator brain-machine interface decoder performance. This platform is developed upon Stable Baselines. 

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).

This platform will make it easier for the research community and industry to develop and evaluate BCI decoders, and will serve as a benchmark to have cross comparison. We expect this platform will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of this platform will allow more researcher to experiment with a more advanced idea, without being buried in monkey experiments. 

We performed four representative decoder algorithms. 
[Placeholder: cite]

## Installation
```
cd BCI_simulator
conda create -n env3.5 python=3.5
conda activate env3.5
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements3.txt
python3 setup.py install
cd gym-centerout
python3 -m pip install -e .
cd ../scripts/
```
## Eample
Run pretrained agents
```
python3 run.py --decoder=hand
```
```
python3 run.py --decoder=FIT
```
```
python3 run.py --decoder=ReFIT
```
```
python3 run.py --decoder=FORCE
```
```
python3 run.py --decoder=VKF
```

Train new agents
```
python3 run.py --decoder=hand --learning_epochs=100 
```
```
python3 run.py --decoder=FIT --learning_epochs=100 --retrain_decoder=1 
```
```
python3 run.py --decoder=ReFIT --learning_epochs=100 --retrain_decoder=1 
```
```
python3 run.py --decoder=FORCE --learning_epochs=100 --retrain_decoder=1 
```
```
python3 run.py --decoder=VKF --learning_epochs=100 --retrain_decoder=1 
```
