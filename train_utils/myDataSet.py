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
                single_pic_bndbox_size = []
                single_pic_object_name = []
                for size in root.find('size'):
                    single_picture_size.append(float(size.text))      #picture format is 'WHC',but we need format is 'CHW' for conv
                single_picture_size[0] , single_picture_size[2] = single_picture_size[2] , single_picture_size[0]   #'WHC' -> 'CHW'
                self.picture_size.append(single_picture_size)
                """collect bndbox size"""
                for object in root.findall('object'):
                    for bndbox in object.findall('bndbox'):
                        for size in bndbox:
                            single_bndbox_size.append(float(size.text))       #xmin,ymin,xmax,ymax
                        single_pic_bndbox_size.append(single_bndbox_size)
                        single_bndbox_size = []
                    """object class name"""
                    single_pic_object_name.append(object.findtext('name'))
            self.object_name.append(single_pic_object_name)
            self.bndbox_size.append([single_pic_bndbox_size])
        assert len(self.object_name) == len(self.bndbox_size) , 'Len error'

    def __getitem__(self,index):# index here means that choose one number in 0~len(self.target['bojectName'])
        index -= 1
        img_path = os.path.join(self.img_path,self.images_path_list[index])
        img = Image.open(img_path,'r')
        target = {'picIndex':None,'picSize':None,'bndboxSize':None,'objectName':None,}
        objectName = self.get_labels(self.object_name[index])
        # use encoding method
        if (self.enconding_method == 'one_hot'):
            objectName = self.one_hot_encoding(objectName)
            objectName = torch.as_tensor(objectName)
        elif (self.enconding_method == None):
            objectName = self.enum_encoding(objectName)

        target['picIndex'] = str(self.picture_index[index])
        target['picSize'] = torch.as_tensor(self.picture_size[index])
        target['bndboxSize'] = torch.as_tensor(self.bndbox_size[index])
        target['objectName'] = objectName

        if self.transform != None:
            img = self.transform(img)
        return img , target

    def __len__(self):
        print(len(self.images_path_list))
        return len(self.images_path_list)

    def get_labels(self,label_name:list):
        labels = []
        with open(os.path.join(os.getcwd(),'labels_encoding.xml'),'rb') as f:
            labels_xml = ET.parse(f)
            root = labels_xml.getroot()
            for label in label_name:
                label = root.find(label)
                labels.append(label.text)
        return labels

    def one_hot_encoding(self,objectName:list):
        labels_sequence = [0 for _ in range(20)]
        output = []
        for object in objectName:
            labels_sequence[int(object)-1] = 1
            output.append(labels_sequence)
            labels_sequence[int(object)-1] = 0
        return output
    def enum_encoding(self,objectName:list):
        return objectName


    # customize dataloader collate_fn method
    @staticmethod
    def collate_fn(batch):          # staticmethod can accept parms without self or cls
        # print(type(batch[0]))
        # print(batch[0][1]['picSize'].size())

        return
