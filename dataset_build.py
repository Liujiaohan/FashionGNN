import os.path as osp
import os
import torch
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T

import numpy as np
import json
import pickle
import random

class PloyvoreDataset(Dataset):
  def __init__(self, root, name, transform = None, pre_transform= None):
    self.name = name
    self.images_path = osp.join(root, name, 'raw', 'images')
    self.image_feature_path = osp.join(root, name, 'raw', 'polyvore_image_vectors')
    self.text_feature_path = osp.join(root, name, 'raw', 'polyvore_text_onehot_vectors')

    # 每个种类所包含的商品
    category_summarize_100 = open(osp.join(root, name, 'raw','category_summarize_100.json'), 'r')
    self.category_summarize_100 = json.load(category_summarize_100)
    # 种类id的index映射
    cid2rcid_100 = open(osp.join(root, name, 'raw','cid2rcid_100.json'), 'r')
    self.cid2rcid = json.load(cid2rcid_100)
    # 获取训练样本
    ftrain = open(osp.join(root, name, 'raw','filtered_training_outfits_processed.json'), 'r')
    self.train_outfit_list = json.load(ftrain)

    super(PloyvoreDataset, self).__init__(root, transform, pre_transform)
 
  @property
  def raw_dir(self):
    return osp.join(self.root, self.name, 'raw')

  @property
  def raw_file_names(self):
    raw_path = osp.join(self.root, self.name, 'raw') 
    return os.listdir(raw_path)

  @property
  def processed_dir(self):
    return osp.join(self.root, self.name, 'processed') 

  @property
  def processed_file_names(self):
    training_size = len(self.train_outfit_list)
    #print ('training_size:{}'.format(training_size * 2))
    processed_file_names = ['data_{}.pt'.format(i) for i in range(training_size*2)] 
    return processed_file_names 
  
  def len(self):
    return len(self.processed_file_names)

  def get(self, idx):
    data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
    return data 

  def process(self): 
    idx = 0
    for outfit in self.train_outfit_list:
      postive_data, negative_data = self.generate_outfit_data(outfit)

      if self.pre_filter is not None and not self.pre_filter(postive_data):
        continue 
      if self.pre_transform is not None:
        postive_data = self.pre_transform(postive_data)
      torch.save(postive_data, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
      idx += 1

      if self.pre_filter is not None and not self.pre_filter(negative_data):
        continue 
      if self.pre_transform is not None:
        negative_data = self.pre_transform(negative_data)
      torch.save(negative_data, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
      idx += 1  

  def generate_outfit_data(self, outfit):
    items_index = outfit['items_index']
    items_category = outfit['items_category']
    outfit_id = outfit['set_id']
    x_pos = []
    x_text_pos = []
    x_neg = []
    x_text_neg = []
    for item_index in items_index: 
      x_pos.append(self.get_img_vector(outfit_id, item_index))
      x_text_pos.append(self.get_text_vector(outfit_id, item_index))
    for category in items_category:
      for category_summarize in self.category_summarize_100:
        if category == category_summarize['id']:
          item = random.choice(category_summarize['items'])
          outfit_id, item_index = item.split("_")
          x_neg.append(self.get_img_vector(outfit_id, item_index))  
          x_text_neg.append(self.get_text_vector(outfit_id, item_index))
    edge_index = []
    for i in range(len(items_index)):
      for j in range(len(items_index)):
        if i != j:
          edge_index.append([i,j])
    edge_index = np.array(edge_index).T
    y_pos = [1]
    y_neg = [0]
    postive_data = Data(x=torch.FloatTensor(x_pos), y=torch.LongTensor(y_pos), edge_index = torch.LongTensor(edge_index))
    postive_data.x_text = torch.FloatTensor(x_text_pos)
    print(postive_data.x_text)
    negative_data = Data(x=torch.FloatTensor(x_neg), y=torch.LongTensor(y_neg), edge_index = torch.LongTensor(edge_index)) 
    negative_data.x_text = torch.FloatTensor(x_text_neg) 
    return postive_data, negative_data
   
  def get_img_vector(self, outfit_id, item_id):
    image_feature = json.load(open(self.image_feature_path + '/'+ str(outfit_id)+ '_' + str(item_id)+ '.json'))
    return image_feature 

  def get_text_vector(self, outfit_id, item_id):
    text_feature = json.load(open(self.text_feature_path + '/' + str(outfit_id) + '_' + str(item_id) + '.json'))
    return text_feature

if __name__ == '__main__':
  dataset = 'Polyvore'
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
  dataset = PloyvoreDataset(path, dataset, T.NormalizeFeatures())
  print (dataset[0], dataset[1]) 
  