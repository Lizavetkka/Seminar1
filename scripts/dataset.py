from torch.utils.data import DataLoader
from torchvision import datasets


def get_data_loaders(batch_size,transform):

    train_data = datasets.FashionMNIST(root = 'data', 
    train= True,
    download= False,
    transform = transform,)

    test_data = datasets.FashionMNIST(root = 'data', 
    train= False,
    download= False,
    transform = transform,)

    train_dataloader = DataLoader(train_data,batch_size = batch_size)
    test_dataloader = DataLoader(test_data,batch_size=batch_size)

    return train_dataloader, test_dataloader