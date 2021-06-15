import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda")

class cycnet(nn.Module):
    def __init__(self, conf, device):
        super(cycnet, self).__init__()
        self.clip_tanh = 10
        self.num_permu = conf["para"]["num_permu"]
        self.v_size = conf["data"]['v_size']
        self.e_size = conf["data"]['e_size']
        self.l_size = self.e_size // self.v_size
        self.mask_e = (torch.ones(self.l_size, self.l_size) -
                       torch.eye(self.l_size)).to(device)
        self.total_e_size = self.num_permu * self.e_size
        self.perma = conf["data"]["perma"]

        self.oddw_v1 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e1 = nn.Parameter(torch.randn(self.l_size, self.l_size))

        self.oddw_v2 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e2 = nn.Parameter(torch.randn(self.l_size, self.l_size))

        self.oddw_v3 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e3 = nn.Parameter(torch.randn(self.l_size, self.l_size))

        self.oddw_v4 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e4 = nn.Parameter(torch.randn(self.l_size, self.l_size))

        self.oddw_v5 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e5 = nn.Parameter(torch.randn(self.l_size, self.l_size))

        self.w_e_out = nn.Parameter(torch.randn(self.l_size))
        self.train_message = torch.zeros(
            conf["para"]["train_batch_size"],self.total_e_size).to(device)
        self.test_message = torch.zeros(
            conf["para"]["test_batch_size"], self.total_e_size).to(device)
        # To generate permutations_rowtocol, permutations_coltorow
        H = np.loadtxt(conf["data"]["H_path"])

        count = 0
        pos_cyclic_col = np.zeros([self.v_size, self.v_size])
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[(j+i) % self.v_size][i] == 1:
                    pos_cyclic_col[(j+i) % self.v_size, i] = count
                    count = count + 1
        count = 0
        pos_cyclic_row = np.zeros([self.v_size, self.v_size])
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[i][(j+i) % self.v_size] == 1:
                    pos_cyclic_row[i, (j+i) % self.v_size] = count
                    count = count + 1

        cycrowtocyccol = np.zeros(self.e_size)
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[i][j] == 1:
                    cycrowtocyccol[np.int(
                        pos_cyclic_col[i][j])] = pos_cyclic_row[i][j]

        cyccoltocycrow = np.zeros(self.e_size)
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[i][j] == 1:
                    cyccoltocycrow[np.int(
                        pos_cyclic_row[i][j])] = pos_cyclic_col[i][j]

        self.permutations_cycrowtocyccol = torch.tensor(np.zeros([len(cycrowtocyccol), len(cycrowtocyccol)])).to(device)
        count = 0
        for j in cycrowtocyccol:
            self.permutations_cycrowtocyccol[np.int(j)][count] = 1
            count = count + 1

        self.permutations_cyccoltocycrow = torch.tensor(np.zeros([len(cyccoltocycrow), len(cyccoltocycrow)])).to(device)
        count = 0
        for j in cyccoltocycrow:
            self.permutations_cyccoltocycrow[np.int(j)][count] = 1
            count = count + 1
        
        self.permutation = torch.zeros([self.num_permu,self.v_size+1,self.v_size+1]).to(device)
        for i in range(self.num_permu):
            permutation_i = torch.zeros([self.v_size+1,self.v_size+1])
            count = 0
            for k in self.perma[i]:
                permutation_i[int(k)][count] = 1
                count = count + 1
            self.permutation[i,:,:] = permutation_i


    def odd_layer(self, inputs_v, inputs_e, oddw_v, oddw_e):
        inputs_v = inputs_v.unsqueeze(2).to(torch.float)
        # batch_size * v_size * l_size = (batch_size * v_size * 1) * ( 1 * l )
        v_out_0 = torch.matmul(inputs_v, oddw_v)
        # inputs_v count by column  b*e = b*v*l
        v_out = torch.zeros(len(inputs_v),self.num_permu,self.v_size,self.l_size).to(device)
        for i in range(int(self.num_permu)):
            v_out_temp = torch.matmul(v_out_0.transpose(1,2), self.permutation[i].to(device)).transpose(1,2)
            v_out[:,i,:,:] = v_out_temp[:,1:,:]
        # inputs_v count by column  b*e = b*v*l 
        v_out = v_out.reshape(-1, self.total_e_size)

        e_intra = torch.tensor([]).to(device)
        for i in range(int(self.num_permu)):
            inputs_e_i = inputs_e[:,i*self.e_size:self.e_size*(i+1)]
            #To do cycrow to : b * e_size = (b * e_size) * (e_size * e*size)
            inputs_e_i = torch.matmul(inputs_e_i, self.permutations_cycrowtocyccol.to(torch.float))
            # b*e = b*v*l * l*l
            mask_w_e = torch.mul(oddw_e, self.mask_e)
            inputs_e_i = inputs_e_i.view(-1, self.v_size, self.l_size)
            e_intra_i = torch.matmul(inputs_e_i,mask_w_e)
            e_intra_i = e_intra_i.view(-1, self.e_size)
            e_intra = torch.cat((e_intra, e_intra_i), 1)
        # add v_out and e_intra
        odd = v_out + e_intra
        odd = 0.5 * torch.clamp(odd, min=-self.clip_tanh, max=self.clip_tanh)
        odd = torch.tanh(odd)
        return odd

    def even_layer(self, odd, flag_clip):
        even = torch.tensor([]).to(device)
        for i in range(int(self.num_permu)):
            odd_i = odd[:,i*self.e_size:self.e_size*(i+1)]
            #To do column to row
            even_i = torch.matmul(odd_i, self.permutations_cyccoltocycrow.to(torch.float))
            # cumulative product then divide itself
            even_i = even_i.view(-1, self.v_size, self.l_size)
            # Matrix value:0->1
            even_i = torch.add(even_i, 1 - (torch.abs(even_i) > 0).to(torch.float))
            prod_even_i = torch.prod(even_i, -1)
            even_i = torch.div(prod_even_i.unsqueeze(2).repeat(1, 1, self.l_size), even_i).reshape(-1, self.e_size)
            if flag_clip:
                even_i = torch.clamp(even_i, min=-self.clip_tanh, max=self.clip_tanh)
            even_i = torch.log(torch.div(1 + even_i, 1 - even_i+0.000000001)+0.000000001)
            even = torch.cat((even,even_i),1)
        return even

    def output_layer(self, inputs_v, inputs_e):
        e_out = torch.tensor([]).to(device)
        for i in range(int(self.num_permu)):
            inputs_e_i = inputs_e[:,i*self.e_size:self.e_size*(i+1)]
            out_layer1 = torch.matmul(inputs_e_i.to(torch.float), self.permutations_cycrowtocyccol.to(torch.float))
            out_layer3 = out_layer1.view(-1,self.v_size,self.l_size)
            # b*v = (b*v*l) * (l)
            e_out_i = torch.matmul(out_layer3, self.w_e_out)
            e_out_add = torch.zeros(len(inputs_v),1).to(device)
            e_out_i = torch.cat((e_out_add,e_out_i),1)
            e_out = torch.cat((e_out,e_out_i),1)
        e_out = e_out.view(-1,self.num_permu,self.v_size+1)
        total_e_out = torch.zeros(len(inputs_v),self.v_size+1).to(device)
        for i in range(int(self.num_permu)):
            # b * (v + 1)
            e_out_i = e_out[:,i,:]
            e_out_i = torch.matmul(e_out_i,self.permutation[i]).to(device)
            total_e_out = total_e_out + e_out_i
        v_out = inputs_v.to(torch.float)
        return v_out + total_e_out

    def forward(self, x, is_train=True):
        flag_clip = 1
        if is_train:
            message = self.train_message
        else:
            message = self.test_message
        odd_result = self.odd_layer(x, message,self.oddw_v1,self.oddw_e1)
        even_result1 = self.even_layer(odd_result, flag_clip)

        flag_clip = 0
        odd_result = self.odd_layer(x, even_result1,self.oddw_v2,self.oddw_e2)
        even_result2 = self.even_layer(odd_result, flag_clip)

        odd_result = self.odd_layer(x, even_result2,self.oddw_v3,self.oddw_e3)
        even_result3 = self.even_layer(odd_result, flag_clip)

        odd_result = self.odd_layer(x, even_result3,self.oddw_v4,self.oddw_e4)
        even_result4 = self.even_layer(odd_result, flag_clip)

        odd_result = self.odd_layer(x, even_result4,self.oddw_v5,self.oddw_e5)
        even_result5 = self.even_layer(odd_result, flag_clip)

        output = self.output_layer(x, even_result5)

        return output