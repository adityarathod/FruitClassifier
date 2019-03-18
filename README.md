# FruitClassifier

A reimplementation of [ArXiv paper 1712.00580](https://arxiv.org/abs/1712.00580) using TF 2.0's high level APIs.

## Todo
- Implement model saving
- Implement test set metrics

## Get Started
*Requires TF 2.0 and Python 3.*
1. Download the [Fruits360 dataset](https://data.mendeley.com/datasets/rp73yg93n8/1).
2. Place Training and Test folders within `data/`.
3. `pip install -r requirements.txt`
4. `cd model && python train.py`
5. Tensorboard logs will be under `logs/`.



## TF 2.0 APIs Used
- New `tf.data` API
    - lazy fetched dataset!
- Improved `tf.keras` API


## Network Structure
*(pulled from the original paper)*

<p align="center">
    <img width="70%" height="auto" src="./network-diagram.png">
</p>


## Disclaimer
I did not write the paper or create the dataset. Most of the work
was done by the paper authors. I implemented their network
in TF 2.0 using the Keras API as a learning exercise.