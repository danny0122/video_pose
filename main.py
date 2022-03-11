import torch
import argparse
import numpy as np
import sys
from tqdm import tqdm


from torch.utils.data import DataLoader

import data.h36m_read
import lib.dataset.h36m_dataset


if __name__ == '__main__':

    print("main : reading h36m data")

    h36m_dataset=lib.dataset.h36m_dataset.H36MDataset(False)

    h36m_dataloader=DataLoader(h36m_dataset, batch_size=32, shuffle=True)

    for i,batch in enumerate(tqdm(h36m_dataloader)):
        import pdb; pdb.set_trace()
        print(batch["img"].shape)

    
    
    #data.h36m_read.load_data("data/Human36M",False)
    #data.h36m_read.read_data_train("data/Human36M","test")
    