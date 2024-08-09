import torch.utils.data as data
import torch
from functools import partial
import os
import pytorch_lightning as pl
from additional_blocks_for_test import Sorter
CHECKPOINT_PATH = "./model_chkpt"

class SortingDataset(data.Dataset):

    def __init__(self,num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories,size=(self.size, self.seq_len))

    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        input_data = self.data[idx]
        labels,_ = torch.sort(input_data)
        return input_data,labels
    
def train_sort(**kwargs):

    root_dir = os.path.join(CHECKPOINT_PATH, "Sorting")
    os.makedirs(root_dir,exist_ok=True)
    train_loader,val_loader,test_loader = kwargs.pop("train_loader"), kwargs.pop("val_loader"),kwargs.pop('test_loader')

    trainer = pl.Trainer(
        default_root_dir = root_dir,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices = 1,
        max_epochs = 10,
        gradient_clip_val = 5, 
    )

    model = Sorter(**kwargs)
    trainer.fit(model, train_loader,val_loader)

    val_result = trainer.test(model, val_loader,verbose=False)
    test_result = trainer.test(model, test_loader,verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":

    dataset = partial(SortingDataset, 100, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = data.DataLoader(dataset(1000), batch_size=128)
    test_loader  = data.DataLoader(dataset(10000), batch_size=128)

    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)

    reverse_model, reverse_result = train_sort(input_dim=train_loader.dataset.num_categories,
                                              model_dim=64,
                                              num_heads=2,
                                              num_classes=train_loader.dataset.num_categories,
                                              num_layers=4,
                                              dropout=0.1,
                                              lr=1e-3,
                                              max_iters=10 * len(train_loader),
                                              train_loader=train_loader,
                                              val_loader= val_loader,
                                              test_loader=test_loader)
    
    print(f"Val accuracy:  {(100.0 * reverse_result['val_acc']):4.2f}%")
    print(f"Test accuracy: {(100.0 * reverse_result['test_acc']):4.2f}%")

    



