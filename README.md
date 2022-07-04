# Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning
This is a Pytorch implementation of the **REDQ+AdaptiveBC** method proposed in the paper "Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning" by Yi Zhao, Rinu Boney, Alexander Ilin, Juho Kannala, and Joni Pajarinen.
## Setup
``` shell
conda env create -f environment.yaml
conda activate adaptive
```
Note: For d4rl, the MuJoCo license is still needed, you can get the license for free from http://roboti.us/, and copy it into your MuJoCo installation foler. The instruction of the [mujoco_py](https://github.com/openai/mujoco-py) might be useful.

## Run the code
The training includes two stages: pretraining on the d4rl dataset and finetuning on the corresponding task. The run the experiment:
```python
python3 main.py --env=<TASK_NAME> --seed=<SEED>
```
We use [wandb](https://wandb.ai/site) for logging. Please check the [documentation](https://docs.wandb.ai/quickstart) for details.