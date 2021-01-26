from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset).__init__()
        self.data = list(range(10))

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=3)
    data = iter(dataloader)
    for i in range(5):
        x = data.__next__()
        print(x)
