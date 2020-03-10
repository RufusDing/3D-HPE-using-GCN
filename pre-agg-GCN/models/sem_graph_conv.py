from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, convm, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convm = convm

        #self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        
        #no sharing: every neighbor has different weight non-interchangeable
        #self.W = nn.Parameter(torch.zeros(size=(62, in_features, out_features), dtype=torch.float))

        #Post-aggregation sharing: every nodes(joints) has different weight but with same weight to all its neighborhood
        self.W = nn.Parameter(torch.zeros(size=(34, in_features, out_features), dtype=torch.float))

        #Convolution-style sharing: only three kinds of weight according to the center of body
        #self.W = nn.Parameter(torch.zeros(size=(3, in_features, out_features), dtype=torch.float))

        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        ''' 
        #adding global feature we need 3 extra weights, for post-aggregation:
        self.Wt = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float)) 
        nn.init.xavier_uniform_(self.Wt.data, gain=1.414)

        #below are using for update the global feature
        self.Wr = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float)) 
        nn.init.xavier_uniform_(self.Wr.data, gain=1.414)

        self.Ws = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float)) 
        nn.init.xavier_uniform_(self.Ws.data, gain=1.414)

        '''

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))   
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        #h0 = torch.matmul(input, self.W[0])
        #h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)
        ###post-agg   pre-agg
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        op = adj * M
        ed = adj * (1 - M)
        ###post-agg

        #output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        #output = torch.zeros(size=(input.shape[0],17, self.out_features), dtype=torch.float).to(input.device)

        h0 = torch.zeros(size=(input.shape[0],17, self.out_features), dtype=torch.float).to(input.device)
        h1 = torch.zeros(size=(input.shape[0],17, self.out_features), dtype=torch.float).to(input.device)
        
        #pre-agg
        #print(input.size())
        for i in range(17):
            h0[:,i,:] = torch.matmul(input[:,i,:], self.W[i])
            h1[:,i,:] = torch.matmul(input[:,i,:], self.W[i+17])
        output = torch.matmul(op, h0) + torch.matmul(ed, h1)
        #print(adj)

        '''
        #post-agg
        for i in range(17):
            h0 = torch.matmul(input, self.W[i])
            h1 = torch.matmul(input, self.W[i+17])
            output[:,i,:] = torch.matmul(op[i], h0) + torch.matmul(ed[i], h1)
        #print(h0.size())
        '''

        '''#conv
        convm = self.convm.to(input.device)
        convm = convm*adj
        #convm[1] = convm[1]*adj
        #convm[2] = convm[2]*adj
        
        for j in range(3):          #0 root    1 near     -1 far
            h0 = torch.matmul(input, self.W[j])
            output += torch.matmul(convm[j], h0)    # using add
        '''

        '''
        #adding global feature
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1) + torch.matmul(self.Wt, input_g)
        #need to think about when to update the global feature? and where?
        
        '''

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
