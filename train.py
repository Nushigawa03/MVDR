import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import argparse

import models

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d','--ddir', default='data/L3DAS_Task1_dev')
parser.add_argument('-b','--batch_size', type=int, default=4)
parser.add_argument('-n','--every_n_steps', type=int, default=16)
parser.add_argument('-e','--epochs', type=int, default=3)
parser.add_argument('-m','--model', default='model.ckpt')
args = parser.parse_args()

net = models.MVDRBeamformer().to(device)

dataset = models.L3DAS_Task1_A_Dataset(args.ddir)

l=dataset.__len__()
t=l//5*4;v=l-t
batch_size=args.batch_size
every_n_steps=args.every_n_steps
print(t,v,batch_size)

train, val = random_split(dataset, [t, v])
datat=DataLoader(
    train,
    batch_size=batch_size,
    collate_fn=models.generate_batch,
    # shuffle=True,
)
datav=DataLoader(
    val,
    batch_size=batch_size,
    collate_fn=models.generate_batch,
    # shuffle=True,
)

checkpoint_dir = 'train'
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", 
    dirpath=checkpoint_dir,
    filename="test-{epoch:04d}",
    every_n_train_steps=every_n_steps,
    save_top_k=3,
    mode="min",
)
loggerTB=pl.loggers.TensorBoardLogger(checkpoint_dir+'/tb_logs', name='my_model')

trainer= pl.Trainer(max_epochs=args.epochs,
                    log_every_n_steps=every_n_steps,
                    logger=loggerTB,
                    callbacks=[checkpoint_callback],
                    gpus=1,
                    limit_train_batches=8,
                    limit_val_batches=4,
                    # fast_dev_run=True,
                    )
trainer.fit(net, datat, datav)
torch.save(net.state_dict(), 'train/'+args.model)


# if __name__ == '__main__':
#     main()