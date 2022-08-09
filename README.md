# MVT: Multi-view Vision Transformer for 3D Object Recognition

## Overview

This an open-source implementation of the work achieved by [Chen et al.](https://arxiv.org/abs/2110.13083)


## How to install dependencies

To install them, run:

```
pip install -r requirements.txt
```

## How to train the model

First, specify the training configuration and all model/DataModule hyperparameters in their respective .yaml config files under configs/, then run in the root directory
chose between mvt_module and mvt_module_with_back_bone in the target name of the model/mvt.yaml config file to train either model

```
python train.py
```

## How to test the model

After specifying the path to the checkpointed model and its config file (respective target name for example) run in root directory:

```
python test.py
```

## Authors 
Project implemented and maintained by Alexandre Lutt, Boris Burkalo, Mohamed Said Derbel, Ludwig Gräf.