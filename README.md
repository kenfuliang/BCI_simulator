# BCI Simulator

BCI simulator is a platform to simulator brain-machine interface decoder performance. This platform is developed upon Stable Baselines. 

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).

This platform will make it easier for the research community and industry to develop and evaluate BCI decoders, and will serve as a benchmark to have cross comparison. We expect this platform will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of this platform will allow more researcher to experiment with a more advanced idea, without being buried in monkey experiments. 

We performed four representative decoder algorithms. 
[Placeholder: cite]

## Installation
```
conda create -n env3.5 python=3.5
conda activate env3.5
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements3.txt
python3 setup.py install
cd gym-centerout
python3 -m pip install -e .
```
## Eample

Run pretrained agents
```
python3 run.py --decoder=hand --decoder_dt=25 --pretrained_mode=1
```
```
python3 run.py --decoder=FIT --decoder_dt=50 --pretrained_mode=1
```
```
python3 run.py --decoder=ReFIT --decoder_dt=50 --pretrained_mode=1
```
```
python3 run.py --decoder=FORCE --decoder_dt=25 --pretrained_mode=1 --num_channel=192
```
```
python3 run.py --decoder=VKF --decoder_dt=50 --pretrained_mode=1
```

Train new agents
```
python3 run.py --decoder=hand --decoder_dt=25 --learning_epochs=100 --target_radius=80 --acceptance_window=40 --min_acceptance_window=25 --learning_rate=2.5e-4 --pretrained_mode=0 --max_acceptance_window=120 --zero_coef=3e-3 --smooth_coef=7e-2 --noise_alpha=0
```
```
python3 run.py --decoder=FIT --decoder_dt=50 --learning_epochs=100 --target_radius=80 --acceptance_window=40 --min_acceptance_window=25 --learning_rate=2.5e-4 --pretrained_mode=1 --retrain_decoder=1 --max_acceptance_window=120 --zero_coef=3e-3 --smooth_coef=7e-2 --noise_alpha=0
```
```
python3 run.py --decoder=ReFIT --decoder_dt=50 --learning_epochs=100 --target_radius=80 --acceptance_window=40 --min_acceptance_window=25 --learning_rate=2.5e-4 --pretrained_mode=1 --retrain_decoder=1 --max_acceptance_window=120 --zero_coef=3e-3 --smooth_coef=7e-2 --noise_alpha=0
```
```
python3 run.py --decoder=FORCE --decoder_dt=25 --learning_epochs=100 --target_radius=80 --acceptance_window=40 --min_acceptance_window=25 --learning_rate=2.5e-4 --pretrained_mode=1 --retrain_decoder=1 --max_acceptance_window=120 --zero_coef=3e-3 --smooth_coef=7e-2 --noise_alpha=0 --num_channel=192
```
```
python3 run.py --decoder=VKF --decoder_dt=50 --learning_epochs=100 --target_radius=80 --acceptance_window=40 --min_acceptance_window=25 --learning_rate=2.5e-4 --pretrained_mode=1 --retrain_decoder=1 --max_acceptance_window=120 --zero_coef=3e-3 --smooth_coef=7e-2 --noise_alpha=0
```
