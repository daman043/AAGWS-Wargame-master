# Dataset for Enemy Position Prediction in AAGWS
[DEP: Dataset for Enemy Position Prediction in AAGWS.]

```
@article{daman043,
  title={DEP: Dataset for Enemy Position Prediction in AAGWS},
  author={Liu, Man and Zhang, Hongjun and Hao, Wenning},
  journal={},
  year={2019}
}
```


## Dataset
### **Preprocessing**

The replays description in  /Wargame-master/data/replay_data/data_description

The replays data in /Wargame-master/data/replay_data/urban_terrain

The programe is /home/jsfz/python_compile/Wargame-master/preprocess/preprocess.py

The preprocessed data in /Wargame-master/data/preprocess_data

We have collected 670 replays which contain about 169125 action records under the "urban residential area" scenario in AAGWS. To ensure the quality of the replays in our dataset, we remain the replays satisfying three criteria:

- The competition time of one game must satisfy the criterion: there are 20 stages in each game, and the actual competition stage must not be less than 10.
- The number of actions of each player in the initial stage must satisfy the criterion: the total number of actions of all pieces in the initial stage must not be less than 15.
- The score of each player must satisfy the criterion: the total score of death pieces of each player must not be less than 10.

The first criterion is to drop out the replays of Interrupted game due to software failure or player exit. The second criterion is to drop out the replays producing by the novice who does not know how to operate the pieces. The third criterion is to drop out the replays in which no pieces died, which means there was no confrontation in the game. After preprocessing, we finally get 515 replays which contain about 12438 action records.

change the directory to /Wargame-master/preprocess, and run

```
python preprocess.py
```



## Parsing Replays

change the directory to /Wargame-master/parse_extract_features, and run

```
python parse_extract_feature.py
```

The result will save in /Wargame-master/data/feature_data

### Split dataset

change the directory to /Wargame-master/parse_extract_features, and run

```
python split.py
```

The result will save in /Wargame-master/data/train_val_test



## Experiment

### Model

The networks in /Wargame-master/Baselines/NetType

### Training

change the directory to /Wargame-master/Baselines/EnemyPositionPrediction, and run

```
python train.py --net_name RES_tensor --phrase train
```

The other argument can be change in code.

The train result will be save in /Wargame-master/Baselines/EnemyPositionPrediction/checkpoints

### Testing

python train.py --net_name RES_tensor --phrase train

```
python train.py --net_name RES_tensor --phrase test
```

The other argument can be change in code.

The test result will be save in /Wargame-master/Baselines/EnemyPositionPrediction/checkpoints/RES_tensor/all

### Plot

Plot the training and test result in /Wargame-master/plot, and 'run result_plot.ipynb' with jupyter notebook

