'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (C) 2021, Integrated Intelligence Research Section, ETRI

Update: 2021.05.03.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class PolicyNet(nn.Module):
    """Class for policy network"""
    def __init__(self, emb_size, key_size, item_size, meta_size, 
                 coordi_size, eval_node, num_rnk, use_batch_norm, 
                 use_dropout, zero_prob, use_multimodal,
                 img_feat_size, name='PolicyNet'):
        """
        initialize and declare variables
        """
        super().__init__()
        self._item_size = item_size
        self._emb_size = emb_size
        self._key_size = key_size
        self._meta_size = meta_size
        self._coordi_size = coordi_size
        self._num_rnk = num_rnk
        self._use_dropout = use_dropout
        self._zero_prob = zero_prob
        self._name = name
        buf = eval_node[1:-1].split('][')
        self._num_hid_eval = list(map(int, buf[0].split(',')))
        self._num_hid_rnk = list(map(int, buf[1].split(',')))
        self._num_hid_layer_eval = len(self._num_hid_eval)
        self._num_hid_layer_rnk = len(self._num_hid_rnk)
        
        mlp_eval_list = OrderedDict([])
        num_in = self._emb_size * self._meta_size * self._coordi_size \
                    + self._key_size
        if use_multimodal:
            num_in += img_feat_size            
        self._count_eval = 0
        for i in range(self._num_hid_layer_eval):
            num_out = self._num_hid_eval[i]
            mlp_eval_list.update({ 
                'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
            mlp_eval_list.update({
                'layer%s_relu'%(i+1): nn.ReLU()})
            if use_batch_norm:
                mlp_eval_list.update({
                    'layer%s_bn'%(i+1): nn.BatchNorm1d(num_out)})
            if self._use_dropout:
                mlp_eval_list.update({
                    'layer%s_dropout'%(i+1): nn.Dropout(p=self._zero_prob)})
            self._count_eval += (num_in * num_out + num_out)            
            num_in = num_out
        self._eval_out_node = num_out 
        self._mlp_eval = nn.Sequential(mlp_eval_list) 

        mlp_rnk_list = OrderedDict([])
        num_in = self._eval_out_node * self._num_rnk + self._key_size
        for i in range(self._num_hid_layer_rnk+1):
            if i == self._num_hid_layer_rnk:
                num_out = math.factorial(self._num_rnk)
                mlp_rnk_list.update({ 
                    'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
            else:
                num_out = self._num_hid_rnk[i]
                mlp_rnk_list.update({ 
                    'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
                mlp_rnk_list.update({
                    'layer%s_relu'%(i+1): nn.ReLU()})
                if use_batch_norm:
                    mlp_rnk_list.update({
                    'layer%s_bn'%(i+1): nn.BatchNorm1d(num_out)})
                if self._use_dropout:
                    mlp_rnk_list.update({
                    'layer%s_dropout'%(i+1): nn.Dropout(p=self._zero_prob)})
            self._count_eval += (num_in * num_out + num_out)
            num_in = num_out
        self._mlp_rnk = nn.Sequential(mlp_rnk_list) 

    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates
        """
        crd_and_req = torch.cat((crd, req), 1)
        evl = self._mlp_eval(crd_and_req)
        return evl
    
    def _ranking_coordi(self, in_rnk):
        """
        rank candidates         
        """
        out_rnk = self._mlp_rnk(in_rnk)
        return out_rnk
        
    def forward(self, req, crd):
        """
        build graph for evaluation and ranking         
        """
        crd_tr = torch.transpose(crd, 1, 0)
        for i in range(self._num_rnk):
            crd_eval = self._evaluate_coordi(crd_tr[i], req)
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)
        in_rnk = torch.cat((in_rnk, req), 1)
        out_rnk = self._ranking_coordi(in_rnk)
        return out_rnk
