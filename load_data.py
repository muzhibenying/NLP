import xml.etree.ElementTree as ET 
import numpy as np 
import torch.utils.data as data 
import torch
from model import Net
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

class DatasetByArticle(data.Dataset):
    """
    dataset for the by article
    """
    def __init__(self, datasetpath):
        super(DatasetByArticle, self).__init__()
        self.pki = torch.load(datasetpath)
        self.samples = self.pki['sample']
        self.labels = self.pki['label']

    def __getitem__(self, index):
        str_index = '%07d' % index
        sample = torch.tensor(self.samples[str_index])
        hyperpartisan = self.labels[str_index]
        if hyperpartisan == 'true':
            label = 1
        if hyperpartisan == 'false':
            label = 0
        return sample, label 

    def __len__(self):
        return len(self.samples)

def my_collate(batch):
    seq_batch = []
    label_batch = []
    seq_len = []
    for sample_tuple in batch:
        label_batch.append(sample_tuple[1])
        seq_batch.append(sample_tuple[0])
        seq_len.append(sample_tuple[0].size()[0])
    seq_len.sort(reverse = True)
    padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_batch, batch_first = True)
    #packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths = seq_len, batch_first = True)
    label_batch = torch.tensor(label_batch)
    return padded_seq_batch, label_batch


filename = '/Users/xiaoying/Downloads/NLP/RNN-classification/最后一次/byarticle.pki'
train_set = DatasetByArticle(filename)
train_loader = data.DataLoader(train_set, batch_size = 4, collate_fn = my_collate)
net = Net(96, 96)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-4)
"""for epoch in range(50):
    running_loss = 0
    running_prediction = []
    running_label = []
    for i, (sample, label) in enumerate(train_loader):
        out = net(sample)
        prediction = out[:, -1]
        loss =criterion(out, label)
        running_loss += loss.item()
        running_prediction.extend(prediction.tolist())
        running_label.extend(label.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss = ', running_loss / len(train_set))
    print('auc = ', roc_auc_score(running_label, running_prediction))
    #print('prediction = ', running_prediction)
    #print('label = ', running_label)"""
for epoch in range(100):
    train_loss = []
    train_prediction = []
    train_label = []
    for i, (sample, label) in enumerate(train_loader):
        if i <= 645 / 4 * 0.8:
            out = net(sample)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    val_prediction = []
    val_label = []
    for i, (sample, label) in enumerate(train_loader):
        if i > 645 / 4 * 0.8:
            out = net(sample)
            prediction = out[:, -1] > 0.5
            val_prediction.extend(prediction.tolist())
            val_label.extend(label.tolist())
    val_prediction = np.array(val_prediction)
    val_label = np.array(val_label)
    right = np.sum(val_label == val_prediction)
    print('accuracy = ', right / val_label.shape[0])

    
        
