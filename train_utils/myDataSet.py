from train_utils import *

class myDataSet(Dataset):
    def __init__(self,transform=None):
        """
        DataSet needs to provide a target dic that has:
            1. picture size         -->list
            2. bouding box size     -->list
            3. picture index        -->list
            4. object_name          -->list
        """
        self.transform = transform
        self.targets = {'picIndex':None,'picSize':None,'bndboxSize':None,'objectName':None,}
        super(myDataSet,self).__init__()
        """training file need to put on the same path with VOCdevkit"""
        self.VOCdevkit_path = os.path.join(os.path.abspath(os.getcwd()), 'VOCdevkit','VOC2007')
        self.img_path = os.path.join(self.VOCdevkit_path, 'JPEGImages')
        assert os.path.exists(self.img_path), "Image Path No Found"
        self.images_path_list = os.listdir(self.img_path)
        assert os.path.exists(self.VOCdevkit_path) , "DataSet path is wrong.Please put the VOCdataSet on the same posision with your train.py!"


        picture_index = []
        picture_size = []
        bndbox_size = []
        object_name = []
        path_list = os.listdir(os.path.join(self.VOCdevkit_path,'Annotations'))
        """collect picture index"""
        for img_xml in path_list:                           #delete the file extend
            picture_index.append(img_xml.split('.xml')[0])
        for picture_name in path_list:
            with open(os.path.join(self.VOCdevkit_path,'Annotations',picture_name),'rb') as f:
                """collect picture size"""
                tree = ET.parse(f)
                root = tree.getroot()
                single_picture_size = []
                single_bndbox_size = []
                for size in root.find('size'):
                    single_picture_size.append(size.text)      #picture format is 'WHC',but we need format is 'CHW' for conv
                single_picture_size[0] , single_picture_size[2] = single_picture_size[2] , single_picture_size[0]   #'WHC' -> 'CHW'
                picture_size.append(single_picture_size)
                """collect bndbox size"""
                for object in root.findall('object'):
                    for bndbox in object.findall('bndbox'):
                        for size in bndbox:
                            single_bndbox_size.append(size.text)       #xmin,ymin,xmax,ymax
                        bndbox_size.append(single_bndbox_size)
                        single_bndbox_size = []
                    """object class name"""
                    object_name.append(object.findtext('name'))
        assert len(object_name) == len(bndbox_size)
        self.targets['picIndex'] = picture_index #---> (N,）
        self.targets['picSize'] = picture_size   #---> (N,3）
        self.targets['bndboxSize'] = bndbox_size #---> (N,4）
        self.targets['objectName'] = object_name #---> (N,）

    def __getitem__(self,index):
        img_path = os.path.join(self.img_path,self.images_path_list[index])
        img = Image.open(img_path,'r')
        target = {'picIndex':None,'picSize':None,'bndboxSize':None,'objectName':None,}
        # print([self.targets['picIndex'][index]])
        picture_index = torch.tensor([index])
        picture_size = torch.tensor(self.targets['picSize'][index])
        bndbox_size = torch.tensor(self.targets['bndboxSize'])
        object_name = torch.tensor([self.targets['objectName']])

        target['picIndex'] = picture_index  # ---> (N,）
        target['picSize'] = picture_size  # ---> (N,3）
        target['bndboxSize'] = bndbox_size  # ---> (N,4）
        target['objectName'] = object_name  # ---> (N,）

        if self.transform != None:
            self.transform(target)
            self.transform(img)
        return img , target


    def __len__(self):
        return len(self.targets['objectName'])