from train_utils import *


dataset = myDataSet(transform=ToTensor(),enconding_method='one_hot')
# dataset.__init__()

# img , targe = dataset[0]

dataloader = DataLoader(dataset,batch_size=10,collate_fn=myDataSet.collate_fn)

iter = dataloader.__iter__()
imgs = iter.next()

