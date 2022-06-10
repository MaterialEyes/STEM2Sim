import torch
import numpy as np
import torchvision
import pandas as pd
import bisect
import os
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import time
from heapq import heappop, heappush, heapify
from collections import defaultdict
from sklearn.manifold import TSNE
from distinctipy import distinctipy
import matplotlib.ticker as ticker
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

class STEMData():
    def __init__(self, ref_id=None, path=None, symmetry_table_id=None, icsd_id=None):
        self.ref_id = ref_id
        self.path = path
        self.symmetry_table_id = symmetry_table_id
        self.icsd_id = icsd_id
        
    def update(self, path, annotation):
        self.path = path
        self.ref_id = float(path.split("/")[-1].split("-")[0])
        item = annotation[annotation["ref_id"] == self.ref_id]
        self.icsd_id = int(item.icsd_id.values)
        self.symmetry_table_id = int(item.symmetry_Int_Tables_number.values)
        
    def __call__(self):
        return {
            "ref_id": self.ref_id,
            "icsd_id": self.icsd_id,
            "symmetry_table_id": self.symmetry_table_id,
            "path": self.path
        }
        

class DataPrep():
    def __init__(self, csv_file, root_dir):
        """
        csv_file: path of annotation file
        root_dir: dir of STEM images
        """
        annotation = pd.read_csv(csv_file)
        imglist = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        self.dataInfo(annotation, imglist)
        self.train, self.val = self.split_train_val(annotation, imglist)
        self.dataGrouping()
    
    def dataInfo(self, annotation, imglist):
        self.path2info = defaultdict(list)
        for path in imglist:
            ref_id = float(path.split("/")[-1].split("-")[0])
            item = annotation[annotation["ref_id"] == ref_id]
            self.path2info[path] = [ref_id, int(item.icsd_id.values), int(item.symmetry_Int_Tables_number.values)]
        
    def split_train_val(self, annotation, imglist, verbose=False):
        """
        randomly split by icsd_id
        train: icsd_id.count = 8
        val: others
        return list of STEMData (train, val)
        """
        counts = annotation.groupby("icsd_id").icsd_id.count()
        values = np.stack(list(annotation.groupby("icsd_id").icsd_id.unique())).reshape(-1)
        assert len(counts) == len(values)
        selected_icsd_ids = values[counts==8]
        
        
        train, val = [], []
        
        if verbose:
            with tqdm(imglist, unit="image") as tepoch:
                for path in tepoch:
                    data = STEMData()
                    data.update(path, annotation)
                    if data.icsd_id in selected_icsd_ids:
                        val.append(data)
                    else:
                        train.append(data)
                    tepoch.set_postfix(train=len(train), val=len(val))
        else:
            for path in imglist:
                data = STEMData()
                data.update(path, annotation)
                if data.icsd_id in selected_icsd_ids:
                    val.append(data)
                else:
                    train.append(data)
        return train, val   
    
    def dataGrouping(self, verbose=False):
        
        self.icsd2img = defaultdict(list)
        self.ref2img = defaultdict(list)
        self.symtable2img = defaultdict(list)
        if verbose:
            with tqdm(self.train, unit="data") as tepoch:
                for data in tepoch:
                    self.icsd2img[data.icsd_id].append(data.path)
                    self.ref2img[data.ref_id].append(data.path)
                    self.symtable2img[data.symmetry_table_id].append(data.path)
                    tepoch.set_postfix(icsd=len(self.icsd2img.keys()), \
                                      ref=len(self.ref2img.keys()), \
                                      symmetry_table = len(self.symtable2img.keys()))
        else:
            for data in self.train:
                self.icsd2img[data.icsd_id].append(data.path)
                self.ref2img[data.ref_id].append(data.path)
                self.symtable2img[data.symmetry_table_id].append(data.path)
                
                
class STEMTestdata(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        csv_file: path of annotation file
        root_dir: dir of STEM images
        """
        annotation = pd.read_csv(csv_file)
        imglist = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        self.train, self.val = self.split_train_val(annotation, imglist)
        self.dataGrouping()
        
        
        
    def split_train_val(self, annotation, imglist, verbose=False):
        """
        randomly split by icsd_id
        train: icsd_id.count = 8
        val: others
        return list of STEMData (train, val)
        """
        counts = annotation.groupby("icsd_id").icsd_id.count()
        values = np.stack(list(annotation.groupby("icsd_id").icsd_id.unique())).reshape(-1)
        assert len(counts) == len(values)
        selected_icsd_ids = values[counts==8]
        
        
        train, val = [], []
        
        if verbose:
            with tqdm(imglist, unit="image") as tepoch:
                for path in tepoch:
                    data = STEMData()
                    data.update(path, annotation)
                    if data.icsd_id in selected_icsd_ids:
                        val.append(data)
                    else:
                        train.append(data)
                    tepoch.set_postfix(train=len(train), val=len(val))
        else:
            for path in imglist:
                data = STEMData()
                data.update(path, annotation)
                if data.icsd_id in selected_icsd_ids:
                    val.append(data)
                else:
                    train.append(data)
        return train, val  
    
    def dataGrouping(self, verbose=False):
        
        self.icsd2img = defaultdict(list)
        self.ref2img = defaultdict(list)
        self.symtable2img = defaultdict(list)
        if verbose:
            with tqdm(self.train, unit="data") as tepoch:
                for data in tepoch:
                    self.icsd2img[data.icsd_id].append(data.path)
                    self.ref2img[data.ref_id].append(data.path)
                    self.symtable2img[data.symmetry_table_id].append(data.path)
                    tepoch.set_postfix(icsd=len(self.icsd2img.keys()), \
                                      ref=len(self.ref2img.keys()), \
                                      symmetry_table = len(self.symtable2img.keys()))
        else:
            for data in self.train:
                self.icsd2img[data.icsd_id].append(data.path)
                self.ref2img[data.ref_id].append(data.path)
                self.symtable2img[data.symmetry_table_id].append(data.path)
                
                
class STEMDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        data_prep = DataPrep(csv_file, root_dir)
        self.train, self.val = data_prep.train, data_prep.val
        self.icsd2img = data_prep.icsd2img
        self.ref2img = data_prep.ref2img
        self.symtable2img = data_prep.symtable2img
        self.path2info = data_prep.path2info
        self.transform = transform

        self.keys = list(self.icsd2img.keys())
        self.icsd_ids = {k:idx for idx, k in enumerate(self.icsd2img.keys())}
        self.symtable_ids = {k:idx for idx, k in enumerate(self.symtable2img.keys())}
        
    def __getitem__(self, idx):
        key = self.keys[idx]
        imgs = np.random.choice(self.icsd2img[key], size=2, replace=False)
        labels = []
        for t in imgs:
            ref_id, icsd_id, symtable_id = self.path2info[t]
            labels.append([ref_id, self.icsd_ids[icsd_id], self.symtable_ids[symtable_id]])
#         labels = [self.path2info[t] for t in imgs] # (ref_id, icsd_id, symmetry_table_id)
        imgs = [Image.open(t).convert("RGB") for t in imgs]

        
        sample = {"imgs": imgs, "labels": labels}
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
        
        
        
    def __len__(self):
        return len(self.icsd2img.keys())

        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    

    def __call__(self, sample):
        imgs, labels = sample['imgs'], sample['labels']
        
        
        imgs = [transforms.ToTensor()(t) for t in imgs]
        labels = [torch.tensor(label) for label in labels]

        return {"imgs": imgs, "labels": labels}   
    
class RandomCrop(object):
    
    def __init__(self, size=224):
        self.size = size
        
    def __call__(self, sample):
        imgs, labels = sample['imgs'], sample['labels']
        imgs = [transforms.RandomCrop(size=self.size)(t) for t in imgs]
        
        return {"imgs": imgs, "labels": labels}   
    
def collate_fn(batch):
    
    imgs = [[],[]]
    labels = [[[],[],[]], [[],[],[]]]
    for sample in batch:
        for idx, t in enumerate(sample["imgs"]):
            imgs[idx].append(t)
        
        for idx, tx in enumerate(sample["labels"]):
            for idy, ty in enumerate(tx):
                labels[idx][idy].append(ty)
                
    imgs = torch.cat([torch.stack(t) for t in imgs], dim=0)
    for idx, tx in enumerate(labels):
        for idy, ty in enumerate(tx):
            labels[idx][idy] = torch.stack(ty)
    
    outpt = []
    for idx in range(len(labels[0])):
        outpt.append(torch.cat([labels[0][idx], labels[1][idx]], dim=0))
            
    return {"imgs": imgs, "labels": outpt}  


class STEMEvaluator():
    def __init__(self):
        pass
    
    def encode(self, encoder, data, batch_size=16):
        img_batch = []
        preds = []
        device = next(iter(encoder.parameters())).device
        tepoch = tqdm(data, unit="batch")
        for path in tepoch:
            img = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
            img_batch.append(img)
            
            if len(img_batch) == batch_size:
                img_batch = torch.cat(img_batch, dim=0).to(device)
                pred = torch.flatten(encoder(img_batch),start_dim=1)
                preds.append(pred.data.cpu())
                img_batch = []
        if len(img_batch) > 0:
            img_batch = torch.cat(img_batch, dim=0).to(device)
            pred = torch.flatten(encoder(img_batch),start_dim=1)
            preds.append(pred.data.cpu())
            img_batch = []
        
        preds = torch.cat(preds, dim=0)
        preds_norm = torch.nn.functional.normalize(preds, dim=1)
#         preds = np.concatenate(preds, axis=0)
        
        return preds, preds_norm
    
    def computeAP(self, encoder, data, batch_size=16, true_labels=None):
        """
        encoder: feature encoder
        data: list of path of images
        true_labels: labels for each image
        
        NOTE: top-1 proposal is self
        """
        preds, preds_norm = self.encode(encoder, data, batch_size)
        preds_norm_np = preds_norm.numpy()
        sim_scores = np.dot(preds_norm_np, preds_norm_np.T)
        
        retrieved_index = []
        tepoch = tqdm(range(sim_scores.shape[0]), unit="data")
        for i in tepoch:
            target_ids = np.where(np.array(true_labels) == true_labels[i])[0]
            sorted_index = list(np.argsort(sim_scores[i])[::-1])
            tmp = []
            for idx in target_ids:
                tmp.append(sorted_index.index(idx))
            retrieved_index.append(sorted(tmp))
        retrieved_index_np = np.array(retrieved_index)
        
        posi = np.linspace(1,retrieved_index_np.shape[1]-1, retrieved_index_np.shape[1]-1).reshape(1,-1)
        AP = np.sum(posi/retrieved_index_np[:,1:], axis=1)/(retrieved_index_np.shape[1]-1)
        mAP = np.mean(AP)
        return preds, sim_scores, AP, mAP
        
    
    
    def classificationEvalution(self, model, data, batch_size=16, true_labels=None, k=20):
        """
        data: list of path of images
        labels: list of category ids
        """
        img_batch = []
        preds = []
        device = next(iter(model.parameters())).device
        tepoch = tqdm(data, unit="batch")
        for path in tepoch:
            img = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
            img_batch.append(img)
            
            if len(img_batch) == batch_size:
                img_batch = torch.cat(img_batch, dim=0).to(device)
                _, cls_pred = net(img_batch)
                preds.append(cls_pred.data.cpu().numpy())
                img_batch = []
        if len(img_batch) > 0:
            img_batch = torch.cat(img_batch, dim=0).to(device)
            _, cls_pred = net(img_batch)
            preds.append(cls_pred.data.cpu().numpy())
            img_batch = []
            
        preds = np.concatenate(preds, axis=0)
        
        if true_labels is None:
            return preds
        
class STEMVisualization():
    def __init__(self):
        pass
    
    
    def showManifold(self, features, labels, num_data=240):
        select_features = features[:num_data].numpy()
        select_labels = labels[:num_data]
        embedding = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(select_features)
        plt.figure(figsize=(7,7))
        colors = distinctipy.get_colors((num_data-1)//24+1)
        for idx, v in enumerate(np.unique(select_labels)):
            c = colors[idx]
            plt.scatter(embedding[np.array(select_labels)==v,0], embedding[np.array(select_labels)==v,1], c=c, label=v)
        plt.legend(bbox_to_anchor=(1., 1.), fontsize=15)
        plt.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
        plt.show()
        
        
    
    
    def showExample(self, idx, data, scores, cand_counts = 20, save=None):
    
        plt.figure(figsize=(5,5))
        img = Image.open(data[idx].path).convert("RGB")
        plt.imshow(img)
        plt.title("Query, icsd={}, symmetry_table={}".format(data[idx].icsd_id, data[idx].symmetry_table_id))
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(20,16))
        sorted_index = list(np.argsort(scores[idx])[::-1])
        for i in range(cand_counts):
            plt.subplot(4,5,i+1)
            img = Image.open(data[sorted_index[i]].path).convert("RGB")
            plt.imshow(img)
            plt.axis("off")
            plt.title("{}, {}, {}".format(sorted_index[i], \
                                                                   data[sorted_index[i]].icsd_id, \
                                                                   data[sorted_index[i]].symmetry_table_id))
        plt.show()    
        if save is None:
            pass