
# Devanagari Handwritten Characters Classification with VGG-8

This project trains a VGG-8 model on [Devanagari Handwritten dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) from UCI. Although the model took had relatively smaller architechture, it overfit the data slightly. The final test accuracy the model got was 99.34%.

## Model Used

VGG-8

![VGG-8 Architechture](model_arch/model_arch.png)


## Data Visualisation

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
```


```python
dataset_path = './Devanagari'
types = ['Train', 'Test']

for type_ in types:
    path_ = os.path.join(dataset_path, type_)
    for character in os.listdir(path_):
        character_path = os.path.join(path_, character)
        character_count = len(os.listdir(character_path))
        plt.bar(character.split('_')[-1], character_count)

    plt.title(type_)
    plt.xticks([])
    plt.xlabel('Characters')
    plt.ylabel('Counts')
    plt.show()
```


    
![Train Dataset Distribution](data_visualisation_files/data_visualisation_1_0.png)
    



    
![Test Dataset Distribution](data_visualisation_files/data_visualisation_1_1.png)
    


Therefore, no class imbalance is seen in the dataset.


```python
dataset_path = './Devanagari'
types = ['Train']
examples = []

for type_ in types:
    path_ = os.path.join(dataset_path, type_)
    for character in os.listdir(path_):
        character_path = os.path.join(path_, character)
        character_example = os.listdir(character_path)[0]
        examples.append([os.path.join(character_path, character_example),character.split('_')[-1]])
```


```python
_, axs = plt.subplots(7, 7, figsize=(12, 12))
axs = axs.flatten()
for i in range(len(examples)):
    image = mpimg.imread(examples[i][0])
    axs[i].imshow(image)
    axs[i].set_title(examples[i][1])

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
```


    
![Examples of Each Class](data_visualisation_files/data_visualisation_4_0.png)
    


It can be seen that the 'jha' character in this dataset is an old one.

## Model Training and Evaluation

```python
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import get_dataloaders
from utils.checkpoint import save, restore
from utils.loops import train, evaluate
from utils.logger import Logger
from vgg import VGG
import os
```


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```


```python

hparams = {
    "net": "VGG_8",
    "name": "VGG_8_Devanagari",
    "batch_size": 32,
    "n_epochs": 20,
    "database_path": "./Devanagari",
    "num_classes": 46,
    'num_workers': 0,
    "start_epoch": 0,
    "restore_epoch": None,
    "model_save_dir": './checkpoints/VGG_8_Devanagari',
    "lr": 0.01,
    "drop": 0.2,
    
}
```


```python
trainloader, valloader, testloader = get_dataloaders(
        path=hparams["database_path"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
    )
```


```python
labels = []
path_ = os.path.join(hparams["database_path"], "train")
for character in os.listdir(path_):
    labels.append(character.split('_')[-1])
```


```python
net = VGG(arch=((1, 64), (1, 128), (1, 256), (1, 512), (1, 512)), lr=hparams["lr"], drop=hparams["drop"],num_classes=hparams["num_classes"])

logger = Logger()
if hparams["restore_epoch"]:
    restore(net, logger, hparams)

```


```python
net = net.to(device)

learning_rate = float(hparams["lr"])
scaler = GradScaler()

optimizer = torch.optim.SGD(
    net.parameters(),
    lr=learning_rate,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.0001,
)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.75, patience=5, verbose=True
)

criterion = nn.CrossEntropyLoss()
```


```python
print("Training", hparams["name"], "on", device)

for epoch in range(hparams["start_epoch"], hparams["n_epochs"]):
    acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler)
    logger.loss_train.append(loss_tr)
    logger.acc_train.append(acc_tr)

    acc_v, loss_v, _ = evaluate(net, valloader, criterion)
    logger.loss_val.append(loss_v)
    logger.acc_val.append(acc_v)

    # Update learning rate
    scheduler.step(loss_v)

    print(f"Epoch {epoch + 1:02} Train Accuracy: {acc_tr:2.4}, Val Accuracy: {acc_v:2.6}")

else:
    save(net, logger, hparams, epoch + 1)
```

    Training VGG_8_Devanagari on cuda:0
    Epoch 01 Train Accuracy: 91.55, Val Accuracy: 97.8052
    Epoch 02 Train Accuracy: 98.06, Val Accuracy: 98.5756
    Epoch 03 Train Accuracy: 98.83, Val Accuracy: 98.3285
    Epoch 04 Train Accuracy: 99.23, Val Accuracy: 98.8081
    Epoch 05 Train Accuracy: 99.45, Val Accuracy: 98.75
    Epoch 06 Train Accuracy: 99.65, Val Accuracy: 98.7936
    Epoch 07 Train Accuracy: 99.7, Val Accuracy: 99.0116
    Epoch 08 Train Accuracy: 99.68, Val Accuracy: 99.0262
    Epoch 09 Train Accuracy: 99.76, Val Accuracy: 98.939
    Epoch 10 Train Accuracy: 99.85, Val Accuracy: 99.0698
    Epoch 11 Train Accuracy: 99.94, Val Accuracy: 99.2151
    Epoch 12 Train Accuracy: 99.89, Val Accuracy: 99.2151
    Epoch 00013: reducing learning rate of group 0 to 7.5000e-03.
    Epoch 13 Train Accuracy: 99.93, Val Accuracy: 99.0552
    Epoch 14 Train Accuracy: 99.95, Val Accuracy: 99.157
    Epoch 15 Train Accuracy: 99.99, Val Accuracy: 99.1424
    Epoch 16 Train Accuracy: 100.0, Val Accuracy: 99.3459
    Epoch 17 Train Accuracy: 100.0, Val Accuracy: 99.3895
    Epoch 18 Train Accuracy: 100.0, Val Accuracy: 99.3605
    Epoch 19 Train Accuracy: 100.0, Val Accuracy: 99.3895
    Epoch 20 Train Accuracy: 100.0, Val Accuracy: 99.3605
    


```python
logger.plot(hparams, save=True, show=True)
```


    
![Accuracy Graph](train_files/train_8_0.png)
    



    
![Loss Graph](train_files/train_8_1.png)
    



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from math import ceil

# Calculate performance on test set
accuracy, loss, y_pred = evaluate(net, testloader, criterion)
y_test = testloader.dataset.labels

# Calculate, precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

cm = confusion_matrix(y_test, y_pred)

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm)

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=False)
plt.title('Confusion Matrix\nAccuracy:{0:.3f}'.format(accuracy))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks([])
plt.yticks([])
plt.show()

print(f"Test Loss: {loss:2.6}")
```

    Accuracy: 99.34782608695652
    Precision: 0.99
    Recall: 0.99
    F1-score: 0.99
    


    
![Confusion Matrix and Evaluation Metrics](train_files/train_9_1.png)
    


    Test Loss: 0.0010148
    


```python
incorrect = np.arange(0,len(y_pred))[y_pred != y_test]
incorrect_pred = np.array(y_pred)[incorrect]
incorrect_test = y_test[incorrect]
incorrect_images = testloader.dataset.images[incorrect]

# Plotting
fig,axs = plt.subplots(ceil(len(incorrect_pred)/5), 5, figsize=(15, 2*ceil(len(incorrect_pred)/5)))
axs = axs.flatten()

for i in range(len(axs)):
    axs[i].imshow(incorrect_images[i], cmap='gray')
    axs[i].set_title(f"True: {labels[incorrect_test[i]]}, Pred: {labels[incorrect_pred[i]]}")
    axs[i].axis('off')
plt.tight_layout()
plt.show()

```


    
![Incorrect Predictions](train_files/train_10_0.png)
    
