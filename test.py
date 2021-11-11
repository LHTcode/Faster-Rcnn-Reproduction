from train_utils import *


dataset = myDataSet(transform=ToTensor(),enconding_method='one_hot')
# dataset.__init__()

# img , targe = dataset[0]

dataloader = DataLoader(dataset,batch_size=3,collate_fn=myDataSet.collate_fn)

for data in dataloader:
    img , targe = data
    print(img)
    print(targe)