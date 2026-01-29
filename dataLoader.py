import numpy as np
import random
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
    
def normalize(img):
    img = img.astype('float64')
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def iqr(image):
    for i in range(image.shape[2]):
        boundry1, boundry2 = np.percentile(image[:, :, i], [2, 98])
        image[:, :, i] = np.clip(image[:, :, i], boundry1, boundry2)
    return image



def data_process(dataset, imgs, labels):
    processed_img = []
    for i in range(imgs.shape[0]):
        if dataset == 'sat':
            img = imgs[i]
            img = iqr(img)
            img = normalize(img)
            temp = np.stack([np.pad(img[:, :, c], [(2, 2), (2, 2)], mode='constant') for c in range(4)], axis=2)
        elif dataset == 'lcz':
            img, _ = imgs[i]
            img = iqr(img)
            temp = normalize(img)
        else:
            img = imgs[i]
            img = iqr(img)
            temp = normalize(img)
        processed_img.append(temp)

    (unique, counts) = np.unique(labels, return_counts=True)

    processed_img = np.array(processed_img)
    processed_label = LabelBinarizer().fit_transform(labels)
    return processed_img, processed_label

    
class DataLoader():
    def __init__(self, dataset, label_ratio):
        self.dataset = dataset
        self.label_ratio = label_ratio
        
        if dataset == 'sat':
            data = scipy.io.loadmat('../Data/SAT-6/sat_vit_h.mat')
            train_x = data['train_x']
            train_y = data['train_y']
            valid_x = data['valid_x']
            valid_y = data['valid_y']
            test_x = data['test_x']
            test_y = data['test_y']
        if dataset == 'lcz':
            rawdata = scipy.io.loadmat('../Data/LCZ/data_5fold_5classes.mat')
            data = rawdata['setting0']
            train_x = data['train_x'][0][0]
            train_y = data['train_y'][0][0][0]
            test_x = data['test_x'][0][0]
            test_y = data['test_y'][0][0][0]
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=2000, random_state=42)
        if dataset == 'eurosat':
            rawdata = scipy.io.loadmat('../Data/eurosat.mat')
            images = rawdata['imgs']
            labels = rawdata['labels']
            labels = [x.strip() for x in labels]
            focus = ['Forest', 'Industrial', 'Residential', 'River', 'SeaLake']
            target_images = []
            target_labels = []
            for i in range(len(labels)):
                if labels[i] in focus:
                    target_labels.append(labels[i])
                    target_images.append(images[i])
            target_images = np.array(target_images)
            target_labels = np.array(target_labels)
            images, labels = shuffle(target_images, target_labels, random_state=42)
            train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=42)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=len(test_x), random_state=42)
        if dataset == 'patternet':
            rawdata = scipy.io.loadmat('../Data/patternet.mat')
            images = rawdata['imgs']
            labels = rawdata['labels']
            labels = [x.strip() for x in labels]
            focus = ['overpass', 'closed_road', 'freeway', 'intersection']
            target_images = []
            target_labels = []
            for i in range(len(labels)):
                if labels[i] in focus:
                    target_labels.append(labels[i])
                    target_images.append(images[i])
            target_images = np.array(target_images)
            target_labels = np.array(target_labels)
            images, labels = shuffle(target_images, target_labels, random_state=42)
            train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=42)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=len(test_x), random_state=42)
        
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y        
    def get_categories(self):
        class_name = [str(x).strip() for x in np.unique(self.train_y)]
        return class_name
    def get_data(self): 
        encoder_train_x = self.train_x[int(self.label_ratio * len(self.train_x)):]
        encoder_train_y = self.train_y[int(self.label_ratio * len(self.train_x)):]
        if self.label_ratio!=1:
            encoder_train_x, encoder_train_y = data_process(self.dataset, encoder_train_x, encoder_train_y)

        train_x = self.train_x[:int(self.label_ratio * len(self.train_x))]
        train_y = self.train_y[:int(self.label_ratio * len(self.train_y))]
        train_x, train_y = data_process(self.dataset, train_x, train_y)
        
        valid_x, valid_y = data_process(self.dataset, self.valid_x, self.valid_y)
        
        test_x, test_y = data_process(self.dataset, self.test_x, self.test_y)
        return encoder_train_x, encoder_train_y, train_x, train_y, valid_x, valid_y, test_x, test_y
        
    def get_ori_data(self): 
        train_x = self.train_x[:int(self.label_ratio * len(self.train_x))]
        train_y = self.train_y[:int(self.label_ratio * len(self.train_y))]
        
        valid_x = self.valid_x
        valid_y = self.valid_y
        
        test_x = self.test_x
        test_y = self.test_y
        return train_x, train_y, valid_x, valid_y, test_x, test_y
        