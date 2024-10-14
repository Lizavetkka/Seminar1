import torch

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    model.train()

    train_loss = 0  
    correct = 0 

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()

        train_loss += loss.item() 
        correct += (pred.argmax(1) == y).type(torch.float).sum().item() 

        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = train_loss / len(dataloader) 
    accuracy = correct / size 
    return {'loss': avg_loss, 'accuracy': accuracy}

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0, 0


    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  

    avg_loss = test_loss / num_batches  
    accuracy = correct / size 
    print(f"Test Error: \n Accuracy: {100 * accuracy:>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return {'loss': avg_loss, 'accuracy': accuracy}
