import enum

from train_utils import *

class myDataSet(Dataset):
    def __init__(self,transform=None,enconding_method=None):
        """
        DataSet needs to provide a target dic that has:
            1. picture size         -->list
            2. bouding box size     -->list
            3. picture index        -->list
            4. self.object_name          -->list
        """
        self.enconding_method = enconding_method
        self.transform = transform
        super(myDataSet,self).__init__()
        """training file need to put on the same path with VOCdevkit"""
        self.VOCdevkit_path = os.path.join(os.path.abspath(os.getcwd()), 'VOCdevkit','VOC2007')
        self.img_path = os.path.join(self.VOCdevkit_path, 'JPEGImages')
        assert os.path.exists(self.img_path), "Image Path No Found"
        self.images_path_list = os.listdir(self.img_path)
        assert os.path.exists(self.VOCdevkit_path) , "DataSet path is wrong.Please put the VOCdataSet on the same posision with your train.py!"


        self.picture_index = []
        self.picture_size = []
        self.bndbox_size = []
        self.object_name = []
        path_list = os.listdir(os.path.join(self.VOCdevkit_path,'Annotations'))
        """collect picture index"""
        for img_xml in path_list:                           #delete the file extend
            self.picture_index.append(img_xml.split('.xml')[0])     #split method can split the str into two part
        for picture_name in path_list:
            with open(os.path.join(self.VOCdevkit_path,'Annotations',picture_name),'rb') as f:
                """collect picture size"""
                tree = ET.parse(f)          #ET is ElementTree model,used for xml files.Parse method used to load the xml file and return a object.
                root = tree.getroot()       #getroot() method can load the root label
                single_picture_size = []
                single_bndbox_size = []
                for size in root.find('size'):
                    single_picture_size.append(float(size.text))      #picture format is 'WHC',but we need format is 'CHW' for conv
                single_picture_size[0] , single_picture_size[2] = single_picture_size[2] , single_picture_size[0]   #'WHC' -> 'CHW'
                self.picture_size.append(single_picture_size)
                """collect bndbox size"""
                for object in root.findall('object'):
                    for bndbox in object.findall('bndbox'):
                        for size in bndbox:
                            single_bndbox_size.append(float(size.text))       #xmin,ymin,xmax,ymax
                        self.bndbox_size.append(single_bndbox_size)
                        single_bndbox_size = []
                    """object class name"""
                    self.object_name.append(object.findtext('name'))
        assert len(self.object_name) == len(self.bndbox_size)

    def __getitem__(self,index):#index here means that choose one number in 0~len(self.target['bojectName'])
        img_path = os.path.join(self.img_path,self.images_path_list[index])
        img = Image.open(img_path,'r')
        target = {'picIndex':None,'picSize':None,'bndboxSize':None,'objectName':None,}
        print(self.object_name[index])
        objectName = int(self.get_labels(self.object_name[index]))
        # use encoding method
        if (self.enconding_method == 'one_hot'):
            objectName = self.one_hot_encoding(objectName-1)

        target['picIndex'] = str(self.picture_index[index])  # ---> (N,）
        target['picSize'] = torch.as_tensor(self.picture_size[index])  # ---> (N,3）
        target['bndboxSize'] = torch.as_tensor(self.bndbox_size[index])  # ---> (N,4）
        target['objectName'] = objectName  # ---> (N,）


        if self.transform != None:
            img = self.transform(img)
        return img , target

    def __len__(self):
        return len(self.targets['objectName'])
    def get_labels(self,label_name:str):
        label = None
        with open(os.path.join(os.getcwd(),'labels_encoding.xml'),'rb') as f:
            labels_xml = ET.parse(f)
            root = labels_xml.getroot()
            label = root.find(label_name)
        return label.text

    def one_hot_encoding(self,objectName:int):
        labels_sequence = [0 for i in range(20)]
        labels_sequence[objectName] = 1
        return labels_sequence

    # customize dataloader collate_fn method
    @staticmethod
    def collate_fn(batch):          # staticmethod can accept parms without self or cls
        print(batch)
        return