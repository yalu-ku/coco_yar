import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class COCODataset(Dataset):
    
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.data_path = root_dir
        self.testing = test
        self.data_type = datalist_file.split('_')[1][:2]
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(root_dir,datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
           image = self.transform(image)
        # if self.testing:
        #     return img_name, image, self.label_list[idx]
        
        return image, self.label_list[idx]
    
    def read_json(self,path,file_name):
        data_path = path + '/annotations/' + file_name
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def file_name_changer(self,int_filename):
        filename = str(int_filename)
        l = len(filename)
        tmp = ''
        data_type = '/train2017/'
        for i in range(12-l):
            tmp += '0'
        if self.data_type == 'va':
            data_type = '/val2017/'
        elif self.data_type == 'te':
            data_type = '/test2017/'
        file_name = self.data_path + data_type + tmp + filename + '.jpg'
        return file_name

    def read_labeled_image_list(self,root_dir,datalist_file):
        data = self.read_json(root_dir,datalist_file)
        annotations = data['annotations']
        images = data['images']
        categories = data['categories']
        cat_class = [[i.get('id')-1, i.get('name')] for i in categories]
        cat_list = ['']*90
        cnt = 0
        for i in range(len(cat_class)):
            if cat_class[i][0] != cnt:
                cnt+=1
            cat_list[cnt] = cat_class[i][1]
            cnt += 1
        img_class = np.unique(np.array([[self.file_name_changer(i.get('image_id')), cat_list[i.get('category_id')-1]] for i in annotations]),axis=0)
        return list(img_class[:,0]),list(img_class[:,1])
