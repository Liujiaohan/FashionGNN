import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool, EdgePooling, GATConv, global_mean_pool

import numpy as np
from matplotlib import pyplot as plt 
from dataset_build import PloyvoreDataset

# todo 加载数据集
dataset = 'Polyvore'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
train_dataset = PloyvoreDataset(path, dataset, transform = T.NormalizeFeatures())
test_dataset = PloyvoreDataset(path, dataset, transform = T.NormalizeFeatures())

train_loader = DataLoader(
  train_dataset[:160], batch_size = 1, shuffle = True
)
test_loader = DataLoader(
  test_dataset[160:], batch_size = 1, shuffle = True
)

class FGAT(torch.nn.Module):
  def __init__(self,
              in_channels,
              out_channels,
              beta = 0.3):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.beta = beta

    text_feature_size = 220
    self.conv1_img = GATConv(train_dataset.num_features, 512, heads= 3, dropout=0.2)
    self.conv1_text = GATConv(text_feature_size, text_feature_size // 2 , heads= 3, dropout=0.2)

    self.conv2_img = GATConv(512 * 3, 512, heads= 1, dropout=0.2)
    self.conv2_text = GATConv(text_feature_size // 2 * 3, text_feature_size // 2, heads= 1, dropout=0.2)

    self.conv3_img = GATConv(512, 20, heads = 1, dropout = 0.2)
    self.conv3_text = GATConv(text_feature_size // 2, 20, heads = 1, dropout= 0.2)

    self.conv4 = GATConv(40, 20, heads= 2, dropout=0.2)

    self.pool1= EdgePooling(40)

    #global_mean_pool 

    self.lin1 = Lin(40, 20) 
    self.lin2 = Lin(20, 2)

  def forward(self, data):
    x_img, edge_index, batch, x_text = data.x, data.edge_index, data.batch, data.x_text

    #image conv1 conv2 con3
    x_img = F.dropout(x_img, p = 0.4, training=self.training)
    x_img = F.relu(self.conv1_img(x_img, edge_index, size = x_img.size(0))) 
    x_img = F.dropout(x_img, p = 0.4, training=self.training)
    x_img = F.relu(self.conv2_img(x_img, edge_index, size = x_img.size(0)))
    x_img = F.dropout(x_img, p = 0.4, training=self.training)
    x_img = F.relu(self.conv3_img(x_img, edge_index, size = x_img.size(0))) 

    #text conv1 conv2 conv3
    # out = self.lin1(torch.cat([x1, x2], dim=1)
    x_text = F.dropout(x_text, p = 0.4, training=self.training)
    x_text = F.relu(self.conv1_text(x_text, edge_index, size = x_text.size(0))) 
    x_text = F.dropout(x_text, p = 0.4, training=self.training)
    x_text = F.relu(self.conv2_text(x_text, edge_index, size = x_text.size(0)))
    x_text = F.dropout(x_text, p = 0.4, training=self.training)
    x_text = F.relu(self.conv3_text(x_text, edge_index, size = x_text.size(0)))

    #cat text_feature and img_feature
    x = torch.cat([x_img, x_text], dim = 1) 

    #conv4
    x = F.relu(self.conv4(x, edge_index, size = x.size(0)))

    #pool1 
    x, edge_index_img, batch_img, _ = self.pool1(x, edge_index, batch)

    #global_pool 
    x = global_max_pool(x, batch_img)

    #lin1 lin2
    x = F.relu(self.lin1(x))
    x = self.lin2(x)

    return F.softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FGAT(train_dataset.num_features, 128, 8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0)

def train():
  model.train()
  total_loss = 0
  for data in train_loader:
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    total_loss += loss.item() * data.num_graphs
    optimizer.step()
  return total_loss / len(train_dataset)

def test(loader):
  model.eval()

  correct = 0
  for data in loader:
    data = data.to(device)
    with torch.no_grad():
      out = model(data)
      pred = out.max(1)[1]
    # print ("pred: {}".format(pred))
    # print ("y: {}".format(data.y))
    correct += pred.eq(data.y).sum().item()
  return correct / len(loader.dataset)

epoch_aix = range(0,5)
loss_history = []
acc_history = []
best_acc = 0
for epoch in range(0, 100):
  loss = train()
  if (epoch % 20 == 0):
    test_acc = test(test_loader)
    if (test_acc > best_acc):
      best_acc = test_acc
      torch.save(model.state_dict(), osp.join(osp.dirname(osp.realpath(__file__)), 'model', 'fgnn_epoch_{:d}_acc_{:.4f}.pkl'.format(epoch, best_acc)))
    loss_history.append(loss)
    acc_history.append(test_acc)
    print('Epoch {:d}, Loss: {:.4f}, Test: {:.4f}'.format(epoch, loss, test_acc))

  scheduler.step()

plt.title("FGNN") 
plt.xlabel("epoch") 
plt.ylabel("loss")
plt.subplot(2,  1,  1) 
plt.plot(epoch_aix, loss_history) 
plt.title('Loss')
plt.subplot(2,  1,  2) 
plt.plot(epoch_aix, acc_history) 
plt.title('acc') 
plt.savefig(osp.join(osp.dirname(osp.realpath(__file__)), 'model', 'fgnn_png'), bbox_inches='tight')




