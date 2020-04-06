# dssm
## 数据分析
该数据集及其不平衡, simtrain_to05sts.txt中各个标签的分布如下：
{'0.0': 10098, '1.0': 1, '2.0': 11, '3.0': 21, '4.0': 90, '5.0': 2526, }
直接训练多分类模型，准确率低。因此，可将该问题转换为二分类问题，即标签值>2,标记为1;否则为0
利用DSSM模型，训练过程如下：
## Training
Epoch 0 | Train Loss: 10.547 | Train Acc: 0.697 | SingleTrainTime: 3.689s
Epoch 0 | Test  Loss: 10.393 | Test  Acc: 0.698 | SingleTestTime: 3.824s

Epoch 1 | Train Loss: 6.648 | Train Acc: 0.892 | SingleTrainTime: 4.531s
Epoch 1 | Test  Loss: 6.756 | Test  Acc: 0.890 | SingleTestTime: 4.150s

Epoch 2 | Train Loss: 3.743 | Train Acc: 0.941 | SingleTrainTime: 4.358s
Epoch 2 | Test  Loss: 3.905 | Test  Acc: 0.942 | SingleTestTime: 4.128s

Epoch 3 | Train Loss: 2.389 | Train Acc: 0.958 | SingleTrainTime: 3.540s
Epoch 3 | Test  Loss: 2.506 | Test  Acc: 0.958 | SingleTestTime: 3.649s

Epoch 4 | Train Loss: 1.715 | Train Acc: 0.971 | SingleTrainTime: 3.521s
Epoch 4 | Test  Loss: 1.849 | Test  Acc: 0.971 | SingleTestTime: 3.544s

Epoch 5 | Train Loss: 1.318 | Train Acc: 0.979 | SingleTrainTime: 3.408s
Epoch 5 | Test  Loss: 1.549 | Test  Acc: 0.976 | SingleTestTime: 3.561s

Epoch 6 | Train Loss: 1.067 | Train Acc: 0.982 | SingleTrainTime: 3.428s
Epoch 6 | Test  Loss: 1.354 | Test  Acc: 0.978 | SingleTestTime: 3.668s

Epoch 7 | Train Loss: 0.886 | Train Acc: 0.986 | SingleTrainTime: 3.465s
Epoch 7 | Test  Loss: 1.220 | Test  Acc: 0.977 | SingleTestTime: 3.574s

Epoch 8 | Train Loss: 0.748 | Train Acc: 0.989 | SingleTrainTime: 3.620s
Epoch 8 | Test  Loss: 1.082 | Test  Acc: 0.979 | SingleTestTime: 3.733s

Epoch 9 | Train Loss: 0.644 | Train Acc: 0.990 | SingleTrainTime: 3.461s
Epoch 9 | Test  Loss: 1.004 | Test  Acc: 0.983 | SingleTestTime: 3.757s

Epoch 10 | Train Loss: 0.566 | Train Acc: 0.991 | SingleTrainTime: 3.454s
Epoch 10 | Test  Loss: 0.946 | Test  Acc: 0.981 | SingleTestTime: 3.555s

Epoch 11 | Train Loss: 0.510 | Train Acc: 0.992 | SingleTrainTime: 3.444s
Epoch 11 | Test  Loss: 0.877 | Test  Acc: 0.983 | SingleTestTime: 3.728s

Epoch 12 | Train Loss: 0.454 | Train Acc: 0.993 | SingleTrainTime: 3.521s
Epoch 12 | Test  Loss: 0.787 | Test  Acc: 0.986 | SingleTestTime: 3.576s

Epoch 13 | Train Loss: 0.425 | Train Acc: 0.993 | SingleTrainTime: 3.614s
Epoch 13 | Test  Loss: 0.752 | Test  Acc: 0.986 | SingleTestTime: 3.776s

Epoch 14 | Train Loss: 0.388 | Train Acc: 0.994 | SingleTrainTime: 3.440s
Epoch 14 | Test  Loss: 0.768 | Test  Acc: 0.986 | SingleTestTime: 3.589s

Epoch 15 | Train Loss: 0.366 | Train Acc: 0.994 | SingleTrainTime: 3.445s
Epoch 15 | Test  Loss: 0.793 | Test  Acc: 0.987 | SingleTestTime: 3.541s

Epoch 16 | Train Loss: 0.335 | Train Acc: 0.994 | SingleTrainTime: 3.424s
Epoch 16 | Test  Loss: 0.763 | Test  Acc: 0.987 | SingleTestTime: 3.573s

Epoch 17 | Train Loss: 0.317 | Train Acc: 0.994 | SingleTrainTime: 3.393s
Epoch 17 | Test  Loss: 0.986 | Test  Acc: 0.987 | SingleTestTime: 3.561s

Epoch 18 | Train Loss: 0.301 | Train Acc: 0.995 | SingleTrainTime: 3.715s
Epoch 18 | Test  Loss: 0.986 | Test  Acc: 0.987 | SingleTestTime: 3.744s

Epoch 19 | Train Loss: 0.287 | Train Acc: 0.995 | SingleTrainTime: 3.592s
Epoch 19 | Test  Loss: 0.768 | Test  Acc: 0.987 | SingleTestTime: 3.596s
Model saved in file:  model/dssm.ckpt
## Tensorboard
loss:
![Image text](https://raw.githubusercontent.com/NiBin90/dssm/master/asserts/loss.jpg)
