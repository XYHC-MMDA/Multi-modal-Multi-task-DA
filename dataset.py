from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset).__init__()
        self.data = list(range(10))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    data = enumerate(dataloader)
    print(len(dataloader))
    for i in range(9):
        x = next(data, None)
        if x is None:
            data = enumerate(dataloader)
            x = next(data)
        print(x)
