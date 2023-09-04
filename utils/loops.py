import torch
from torch.cuda.amp import autocast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, dataloader, criterion, optimizer, scaler, scheduler = None):

    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast():
                
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_tr += loss.item()

            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return acc, loss

def evaluate(net, dataloader, criterion):

    net = net.eval()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    total_preds = []

    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = net(inputs)

        loss = criterion(outputs, labels)

        # calculate performance metrics
        loss_tr += loss.item()

        _, preds = torch.max(outputs.data, 1)
        correct_count += (preds == labels).sum().item()
        n_samples += labels.size(0)
        total_preds.extend(preds.tolist())

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return acc, loss, total_preds