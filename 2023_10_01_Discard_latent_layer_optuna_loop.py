## Init

import subprocess
cmd = "egrep '^(NAME)=' /etc/os-release" 

location = str(subprocess.check_output(cmd, shell=True).rstrip().decode('utf-8'))[5:].strip('"')


# if location == "Ubuntu":
#     print(location)
#     prefix_path = "/content/drive/MyDrive/"  
#     # mount drive
#     from google.colab import drive
#     import sys
#     drive.mount('/content/drive')
    
# else:  
#     prefix_path = "/work/home/nunthawutc/"

prefix_path = "/work/home/nunthawutc/"

dataset_path    = prefix_path + "datasets/"
mask_path       = prefix_path + "databases/masks/"
optuna_path     = prefix_path + "optuna_study/pilot/"
weight_path     = prefix_path + "weight_bias/junk/"
figure_path     = prefix_path + "figure_table/"
tensorboard_path= prefix_path + "tensorboard/junk/"



import gc

import os

import joblib

import math

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from collections import Counter

from statistics import mean

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold , StratifiedKFold ,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report ,accuracy_score ,f1_score ,roc_curve ,auc
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.storages import JournalFileStorage
from optuna.samplers import RandomSampler


class CustomDataLoader(DataLoader):
    def __iter__(self):
        data_iter = super().__iter__()
        for batch in data_iter:
            if len(batch) > 1:  # Check if the batch has more than one sample
                yield batch

def KLD_function(mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

class EarlyStopping_without_model_weight:
    ''' version 2.0 '''
    '''     maximize & minimize '''
    '''     enable/disable weight checkpoint '''

    """Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py """
    """Early stops the training if validation loss/accuracy doesn't improve after a given patience."""
    def __init__(self, mode='minimize', patience=15, verbose=False, delta=0, path='checkpoint.pt', save_state=False, trace_func=print):
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_accu_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_epoch_dict = None
        self.save_state = save_state

    def __call__(self, epoch, epoch_dict, model):
        if self.mode == 'minimize':
            score = -epoch_dict['epoch_val_loss']
        elif self.mode == 'maximize':
            score = epoch_dict['epoch_accuracy']

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, epoch_dict, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, epoch_dict, model)
            self.counter = 0

    def save_checkpoint(self, epoch, epoch_dict, model):
        if self.verbose:
            if self.mode == 'minimize':
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {epoch_dict["epoch_val_loss"]:.6f}).  Saving model ...')
            elif self.mode == 'maximize':
                self.trace_func(f'Validation accuracy increased ({self.val_accu_max:.6f} --> {epoch_dict["epoch_accuracy"]:.6f}).  Saving model ...')

        if self.mode == 'minimize':
            self.val_loss_min = epoch_dict['epoch_val_loss']
        elif self.mode == 'maximize':
            self.val_accu_max = epoch_dict['epoch_accuracy']

        self.best_epoch_dict = {'epoch': epoch, 'loss': epoch_dict['epoch_val_loss'], 'accuracy': epoch_dict['epoch_accuracy'], 'f1_score': epoch_dict['epoch_f1_score']}
        if self.save_state:
            torch.save(model.state_dict(), self.path)
            # print("model saved")


def load_weight(model,state_dict):
    with torch.no_grad():
        model.enc1.weight.copy_(state_dict['enc1.weight'])
        model.enc1.bias.copy_(state_dict['enc1.bias'])
        model.enc2.weight.copy_(state_dict['enc2.weight'])
        model.enc2.bias.copy_(state_dict['enc2.bias'])
        model.enc3.weight.copy_(state_dict['enc3.weight'])
        model.enc3.bias.copy_(state_dict['enc3.bias'])
        model.dec1.weight.copy_(state_dict['dec1.weight'])
        model.dec1.bias.copy_(state_dict['dec1.bias'])
        model.dec2.weight.copy_(state_dict['dec2.weight'])
        model.dec2.bias.copy_(state_dict['dec2.bias'])
        model.dec3.weight.copy_(state_dict['dec3.weight'])
        model.dec3.bias.copy_(state_dict['dec3.bias'])
    print("weight & bias loaded !!")

class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]

  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extended torch.nn module which cusmize connection.
This code base on https://pytorch.org/docs/stable/notes/extending.html
"""
import math
import torch
import torch.nn as nn

#################################
# Define custome autograd function for masked connection.

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


if __name__ == 'check grad':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    customlinear = CustomizedLinearFunction.apply

    input = (
            torch.randn(20,20,dtype=torch.double,requires_grad=True),
            torch.randn(30,20,dtype=torch.double,requires_grad=True),
            None,
            None,
            )
    test = gradcheck(customlinear, input, eps=1e-6, atol=1e-4)
    print(test)


def plot_grad_flow(named_parameters):
    # https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    # Check gradient flow in network
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.01) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(figure_path+ "DEEP_AE_epoch"+ str(epoch) +".jpg")



class DatasetMaker:
    def __init__(self, x_df, y_df, held_out_clf, target_gene_symbol, sampling_rate , true_heldout_rate, normalization_type):
        self.x_df = x_df
        self.y_df = y_df
        self.held_out_clf = held_out_clf
        self.target_gene_symbol = target_gene_symbol
        self.sampling_rate = sampling_rate
        self.true_heldout_rate = true_heldout_rate
        self.normalization_type = normalization_type

    def normalize_dataset(self):
        if self.normalization_type == 'no_scaler':
            return self.x_df
        else:
            if self.normalization_type == 'std_scaler':
                scaler = StandardScaler()
            elif self.normalization_type == 'minmax_0_1':
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif self.normalization_type == 'minmax_1_1':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                raise ValueError("Invalid normalization_type.")

            normalized_array = scaler.fit_transform(self.x_df)
            return pd.DataFrame(normalized_array, index=self.x_df.index, columns=self.x_df.columns)

    def create_dataset(self):
        selected = self.y_df[self.y_df.Subtype_Selected.isin(self.held_out_clf)]['Unnamed: 0'].tolist()
        selected_columns = selected

        normed_df = self.normalize_dataset()

        df_selected = normed_df.loc[selected_columns]
        df_remaining = normed_df.drop(index=selected_columns)

        y = self.y_df[self.y_df['Unnamed: 0'].isin(df_selected.index.tolist())]['Subtype_Selected'].tolist()

        if self.true_heldout_rate == 0:  # AE will see every sample including BRCA samples 
            X_train, X_test, y_train, y_test = train_test_split(df_selected, y, test_size=0.4, random_state=42)
            dev_unsup_set = normed_df.sample(frac=self.sampling_rate)
            dev_sup_set = (X_train, y_train)
            test_sup_set = (X_test, y_test)

            return dev_unsup_set, dev_sup_set, test_sup_set

        elif self.true_heldout_rate == 1:   # AE will never see BRCA samples
            X_train, X_test, y_train, y_test = train_test_split(df_selected, y, test_size=0.4, random_state=42)
            dev_unsup_set = df_remaining.sample(frac=self.sampling_rate)
            dev_sup_set = (X_train, y_train)
            test_sup_set = (X_test, y_test)

            return dev_unsup_set, dev_sup_set, test_sup_set

        else:  # normal use case AE will see some sameple from BRCA samples
            X_train, X_test, y_train, y_test = train_test_split(df_selected, y, test_size=self.true_heldout_rate, random_state=42)
            dev_unsup_set = pd.concat([df_remaining, X_train], axis=0)
            dev_unsup_set = dev_unsup_set.sample(frac=self.sampling_rate)
            dev_sup_set = (X_train, y_train)
            test_sup_set = (X_test, y_test)

            return dev_unsup_set, dev_sup_set, test_sup_set


def training(model , optimizer , MSE_function , epochs , trial ,  unsup_train_loader , unsup_val_loader, optuna = True):
    early_stop = EarlyStopping_without_model_weight(patience=2, mode = 'minimize', save_state = False)
    for epoch in range(epochs):
        train_MSE = 0
        train_loss = 0
        test_MSE = 0
        test_loss =0

        for batch_features in unsup_train_loader:
            model.train()
            batch_features = batch_features.to(device)

            optimizer.zero_grad()

            pathway , recon  = model(batch_features)
            MSE = MSE_function(batch_features,recon)

            Loss = MSE
            Loss.backward()
            optimizer.step()

            train_loss += Loss.item()
            train_MSE += MSE.item()


        with torch.no_grad():
            model.eval()
            for batch_features  in unsup_val_loader:
                batch_features = batch_features.to(device)

                pathway, recon  = model(batch_features)
                MSE = MSE_function(batch_features,recon)


                test_MSE += MSE.item()
        # if optuna == True:
        #     trial.report( test_MSE / len(unsup_val_loader) , step=epoch)

        epoch_dict = {'epoch': epoch,
                      'epoch_val_loss': test_MSE / len(unsup_val_loader),
                      'epoch_accuracy': 0,
                      'epoch_f1_score': 0}

        early_stop(epoch, epoch_dict,model)
        if optuna == True:
            if epoch % 1 == 0:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("        " ,current_time, epoch + 1 , train_MSE / len(unsup_train_loader), test_MSE/len(unsup_val_loader))
        if early_stop.early_stop:
            break

    return model


"""

new inference function

feature added : handle the case when model have 3 or 5 layers

"""
def inference(model, layers , dataloader_):
    if layers == 5:
        with torch.no_grad():
            temp_input = torch.empty((0,18303)).to(device)
            temp_pathway = torch.empty((0,2659)).to(device)
            temp_latent = torch.empty((0,200)).to(device)
            temp_recon = torch.empty((0,18303)).to(device)
            temp_label = torch.empty((0,1)).to(device)
            model.eval()
            for batch_features , y in dataloader_:
                batch_features = batch_features.to(device)
                y = y.to(device)
                y = y.unsqueeze(1)

                pathway , latent , recon  = model(batch_features)

                temp_input = torch.cat((temp_input,batch_features))
                temp_pathway = torch.cat((temp_pathway,pathway))
                temp_latent = torch.cat((temp_latent,latent))
                temp_recon = torch.cat((temp_recon,recon))
                temp_label = torch.cat((temp_label,y))

        temp_concat = torch.cat((temp_pathway, temp_latent), dim=1)


        return temp_input, temp_pathway, temp_latent, temp_concat , temp_recon, temp_label

    else:
        with torch.no_grad():
            temp_input = torch.empty((0,18303)).to(device)
            temp_pathway = torch.empty((0,2659)).to(device)
            temp_recon = torch.empty((0,18303)).to(device)
            temp_label = torch.empty((0,1)).to(device)
            model.eval()
            for batch_features , y in dataloader_:
                batch_features = batch_features.to(device)
                y = y.to(device)
                y = y.unsqueeze(1)

                pathway ,  recon  = model(batch_features)

                temp_input = torch.cat((temp_input,batch_features))
                temp_pathway = torch.cat((temp_pathway,pathway))
                temp_recon = torch.cat((temp_recon,recon))
                temp_label = torch.cat((temp_label,y))

        return temp_input, temp_pathway, temp_recon, temp_label

"""
new LR function

Feature removed : change from nested to normal CV
"""

def LR(latent, y , test_latent , test_y):
    latent = latent.cpu().numpy()
    y = y.ravel().cpu().numpy()

    test_latent = test_latent.cpu().numpy()
    test_y = test_y.ravel().cpu().numpy()


    # Define the parameter grids
    param_grid_lr = {'C': [0.001, 0.01 ,0.1, 1 ,10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}

    # Initialize the model
    clf_lr = LogisticRegression(max_iter=1000, verbose =0)



    # Define the inner cross-validation loop for logistic regression
    inner_cv_lr = GridSearchCV(estimator=clf_lr, param_grid=param_grid_lr, cv=5 , verbose = 0)
    inner_cv_lr.fit(latent, y)
    best_lr = inner_cv_lr.best_estimator_

    # print(inner_cv_lr.best_params_)

    # Evaluate the model on the test set in the outer loop
    y_pred_lr = best_lr.predict(test_latent)

    return best_lr.score(test_latent, test_y)

"""

new function for calculating clustering metrics

"""

def Cluster_metrics(test_data, test_label):
    results = {'raw': {'ari': [], 'nmi': [], 'sic': []}, 'reduc': {'ari': [], 'nmi': [], 'sic': []}}

    for fold in range(5):
        feature_ = test_data.cpu()

        for reduc in ['raw', 'reduc']:
            if reduc == 'raw':
                kmeans = KMeans(n_clusters=4, n_init=10)
                cluster_labels = kmeans.fit_predict(feature_)
            if reduc == 'reduc':
                reducer = umap.UMAP()
                X_embedded = reducer.fit_transform(feature_)
                kmeans = KMeans(n_clusters=4, n_init=10)
                cluster_labels = kmeans.fit_predict(X_embedded)

            ari_score = adjusted_rand_score(test_label.ravel().cpu(), cluster_labels)
            nmi_score = normalized_mutual_info_score(test_label.ravel().cpu(), cluster_labels)
            if reduc == 'raw':
                sic_score = silhouette_score(feature_, test_label.ravel().cpu(), metric='euclidean')
            if reduc == 'reduc':
                sic_score = silhouette_score(X_embedded, test_label.ravel().cpu(), metric='euclidean')

            results[reduc]['ari'].append(convert_floats(ari_score))
            results[reduc]['nmi'].append(convert_floats(nmi_score))
            results[reduc]['sic'].append(convert_floats(sic_score))

    return results

def dictlist_conversion(dict_list):
    results = {'raw': {'ari': [], 'nmi': [], 'sic': []}, 'reduc': {'ari': [], 'nmi': [], 'sic': []}}

    for dict_ in dict_list:
        # print(dict_)
        for reduc , metrics in dict_.items():
            for metric , values in metrics.items():
                results[reduc][metric].extend(values)
    # print(results)
    for reduc , metrics in results.items():
        for metric , values in metrics.items():
            results[reduc][metric] = mean(values)
    # print(results)
    return results


def construct_AE(data_scaler_type, layer_type, norm_layer_type, activation_function, layers=5, dense_mask_tensor=None):
    """
    Version 2.
    Features added: Ability to handle selector for multiple AE depths (3 layers or 5 layers).
    """
    if activation_function == 'linear':
        act_fn = lambda x: x  # Linear activation
    elif activation_function == 'relu':
        act_fn = F.relu
    elif activation_function == 'leaky_relu':
        act_fn = F.leaky_relu
    elif activation_function == 'tanh':
        act_fn = F.tanh
    elif activation_function == 'swish':
        act_fn = lambda x: x * F.sigmoid(x)
    elif activation_function == 'elu':
        act_fn = F.elu
    elif activation_function == 'prelu':
        act_fn_module = nn.PReLU()
        act_fn = act_fn_module.forward

    if layer_type == 'fc':
        ENC1 = nn.Linear(18303, 2659)
        DEC2 = nn.Linear(2659, 18303)
    elif layer_type == 'cl':
        ENC1 = CustomizedLinear(dense_mask_tensor, bias=True)
        DEC2 = CustomizedLinear(dense_mask_tensor.T, bias=True)

    # Base Model
    class AE_3Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ENC1 = ENC1
            self.DEC2 = DEC2
            if activation_function == 'prelu':
                self.act_fn = nn.PReLU()
            else:
                self.act_fn = act_fn

        def forward(self, X):
            pathway = self.act_fn(self.ENC1(X))
            recon = self.act_fn(self.DEC2(pathway))
            return pathway, recon

    class AE_5Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ENC1 = ENC1
            self.ENC2 = nn.Linear(2659, 200)
            self.DEC1 = nn.Linear(200, 2659)
            self.DEC2 = DEC2
            if activation_function == 'prelu':
                self.act_fn = nn.PReLU()
            else:
                self.act_fn = act_fn

        def forward(self, X):
            pathway = self.act_fn(self.ENC1(X))
            latent = self.act_fn(self.ENC2(pathway))
            X = self.act_fn(self.DEC1(latent))
            recon = self.act_fn(self.DEC2(X))
            return pathway , latent, recon

# Choose appropriate base model
    AE_Base = AE_5Layer if layers == 5 else AE_3Layer


    # Normalization
    if norm_layer_type == 'none':
        return AE_Base().to(device)
    elif norm_layer_type == 'batch_norm':
        class AE_BN(AE_Base):
            def __init__(self):
                super().__init__()
                self.batch_norm1 = nn.BatchNorm1d(2659)
                if layers == 5:
                    self.batch_norm2 = nn.BatchNorm1d(200)

            def forward(self, X):
                pathway = self.act_fn(self.batch_norm1(self.ENC1(X)))
                if layers == 5:
                    latent = self.act_fn(self.batch_norm2(self.ENC2(pathway)))
                    X = self.act_fn(self.batch_norm1(self.DEC1(latent)))
                    recon = self.act_fn(self.DEC2(X))
                    return pathway, latent, recon
                else:
                    recon = self.act_fn(self.DEC2(pathway))
                    return pathway, recon

        return AE_BN().to(device)

    elif norm_layer_type == 'layer_norm':
        class AE_LN(AE_Base):
            def __init__(self):
                super().__init__()
                self.layer_norm1 = nn.LayerNorm(2659)
                if layers == 5:
                    self.layer_norm2 = nn.LayerNorm(200)

            def forward(self, X):
                pathway = self.act_fn(self.layer_norm1(self.ENC1(X)))
                if layers == 5:
                    latent = self.act_fn(self.layer_norm2(self.ENC2(pathway)))
                    X = self.act_fn(self.layer_norm1(self.DEC1(latent)))
                    recon = self.act_fn(self.DEC2(X))
                    return pathway, latent, recon
                else:
                    recon = self.act_fn(self.DEC2(pathway))
                    return pathway, recon

        return AE_LN().to(device)


def convert_floats(data):
    if isinstance(data, dict):
        return {k: convert_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_floats(v) for v in data]
    elif isinstance(data, float):
        return float(data)
    else:
        return data


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# import mask

mask_path       = prefix_path + "databases/masks/"
#Pickle load
import scipy
import pickle
file = open(mask_path + "connectivity_matrix_TcgaTargetGtex_filterd_2022_06_24.pickle",'rb')
Thaw_connectivity_matrix = pickle.load(file)  # sparse_coo format
dense_mask_tensor = torch.tensor(Thaw_connectivity_matrix.toarray(),dtype=torch.float32)

# dataset

BRCA_label_dict = {'BRCA.Basal': 0,
                    'BRCA.Her2': 1,
                    'BRCA.LumA': 2,
                    'BRCA.LumB': 3}

with open(dataset_path + "TcgaTargetGtex/TcgaTargetGtex_RSEM_Hugo_norm_count_gene_symbol_CGP_intersection_final.txt", "rb") as in_file:
    TcgaTargetGtex_symbol_intersect = pickle.load(in_file)
    
y_df = pd.read_csv(dataset_path + "TcgaTargetGtex/TcgaTargetGTEX_phenotype_plus_subtype_final_(reading_df3).csv")
held_out_clf = ['BRCA.LumA','BRCA.LumB','BRCA.Basal','BRCA.Her2']
x_df = pd.read_feather(dataset_path + "TcgaTargetGtex/TcgaTargetGtex_RSEM_Hugo_norm_count.feather")
x_df = x_df[x_df['sample'].isin(TcgaTargetGtex_symbol_intersect)]
x_df.set_index(['sample'],inplace=True)
x_df = x_df.T
def objective(trial):

    combined_choices = []

    scalers = ['std_scaler', 'minmax_1_1', 'minmax_0_1', 'no_scaler']

    for scaler in scalers:
        if scaler == 'std_scaler':
            activation_functions = [ 'swish', 'elu', 'prelu']
        elif scaler == 'minmax_1_1':
            activation_functions = ['tanh', 'swish', 'elu']
        elif scaler == 'minmax_0_1':
            activation_functions = ['relu', 'elu', 'prelu']
        elif scaler == 'no_scaler':
            activation_functions = ['relu', 'elu', 'prelu']

        for activation in activation_functions:
            combined_choices.append((str(scaler)+ ":"+ str(activation)))

    chosen_combination = trial.suggest_categorical("scaler_with_activation", combined_choices)
    data_scaler_type, activation_function = chosen_combination.split(":")


    sampling_rate = 1.0
    heldout_rate = trial.suggest_categorical("heldout_rate" , [0 , 0.3 , 1.0])
    norm_layer = trial.suggest_categorical("norm_layer" , ["batch_norm" , "layer_norm"])
    layer = trial.suggest_categorical("layer" , ["fc" , "cl"])
    lr = trial.suggest_categorical("lr" , [1e-3 , 5e-4 , 1e-4 , 5e-5])
    layers = 3

    dataset_maker = DatasetMaker(x_df, y_df, held_out_clf, TcgaTargetGtex_symbol_intersect ,sampling_rate , heldout_rate ,  data_scaler_type)
    dev_unsup_set, dev_sup_set, test_sup_set = dataset_maker.create_dataset()



    # ------------ This block helps the AE model to handle stratified K-fold cross-validation -----------
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    unsup_y = []
    subtype_dict = dict(zip(dev_sup_set[0].index.tolist(),dev_sup_set[1]))
    for sample in dev_unsup_set.index.tolist():
        if sample in subtype_dict.keys():
            # y.append(subtype_dict[sample])
            unsup_y.append(BRCA_label_dict[subtype_dict[sample]])
        else:
            unsup_y.append(99)
    unsup_y = np.array(unsup_y)
    # ------------ This block helps the AE model to handle stratified K-fold cross-validation -----------

    test_accuracy_list = []
    pathway_dict_list = []
    recon_dict_list = []

    for train_index, val_index in skf.split(dev_unsup_set.values,unsup_y):
        model = construct_AE(data_scaler_type=data_scaler_type,
                                layer_type=layer,
                                norm_layer_type=norm_layer,
                                activation_function = activation_function,
                                layers = layers,
                                dense_mask_tensor = dense_mask_tensor)

        unsup_train= dev_unsup_set.values[train_index]
        unsup_val = dev_unsup_set.values[val_index]

        unsup_train = torch.tensor(unsup_train,dtype=torch.float32)
        unsup_val = torch.tensor(unsup_val,dtype=torch.float32)

        unsup_train_loader = CustomDataLoader(unsup_train,batch_size=128,shuffle=True)
        unsup_val_loader = CustomDataLoader(unsup_val,batch_size=len(unsup_val),shuffle=True)


        optimizer = optim.Adam(model.parameters(),lr = lr)
        MSE_function = nn.MSELoss()
        epochs = 2000
        model = training(model,optimizer,MSE_function,epochs, trial , unsup_train_loader , unsup_val_loader ,optuna= True)
        model.eval()

        dev_sup_set_y = [BRCA_label_dict[i] for i in dev_sup_set[1]]
        sup_dev_loader = CustomDataLoader(dataset(dev_sup_set[0].values, dev_sup_set_y),batch_size=len(dev_sup_set[1]),shuffle=True)

        sup_y_test = [BRCA_label_dict[i] for i in test_sup_set[1]]
        sup_test_loader = CustomDataLoader(dataset(test_sup_set[0].values, sup_y_test) , batch_size = len(sup_y_test) , shuffle = False)

        dev_input , dev_pathway , dev_recon , dev_label = inference(model ,layers ,sup_dev_loader)
        test_input , test_pathway , test_recon , test_label = inference(model ,layers ,sup_test_loader)


        test_pw_avg_accu =LR(dev_pathway ,dev_label , test_pathway ,test_label)
        test_accuracy_list.append(test_pw_avg_accu )

        pathway_dict_list.append(Cluster_metrics(test_pathway, test_label))
        recon_dict_list.append(Cluster_metrics(test_recon, test_label))


    pathway_dict = dictlist_conversion(pathway_dict_list)
    recon_dict = dictlist_conversion(recon_dict_list)

    flattened_test_pathway = flatten_dict(pathway_dict)
    flattened_test_recon = flatten_dict(recon_dict)

    for key, value in flattened_test_pathway.items():
        attr_name = f"test_pathway_{key}"
        trial.set_user_attr(attr_name, float(value))

    for key, value in flattened_test_recon.items():
        attr_name = f"test_recon_{key}"
        trial.set_user_attr(attr_name, float(value))

    print(test_accuracy_list)
    return sum(test_accuracy_list)/n_splits


# for Journal storage
storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("2023_10_02_Discard_latent_layer_optuna_loop.log"))

study = optuna.create_study(direction="maximize",
                            storage = storage,
                            study_name="2023_10_02_Discard_latent_layer_optuna_loop",
                            sampler=optuna.samplers.BruteForceSampler(),
                            load_if_exists=True)

study.optimize(objective, n_trials=4000 ,callbacks=[MaxTrialsCallback(4000, states=(TrialState.COMPLETE,))])
