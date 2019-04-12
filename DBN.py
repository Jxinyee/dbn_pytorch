import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from RBM import RBM
import time
class DBN(nn.Module):

    def __init__(self,
                 visible_units=41,  # 可视层节点 根据你的特征维度 ，如果有4个feature那就是4 如果有26个那就是26
                 hidden_units=[11, 6, 11],  # 隐藏层节点
                 k=5,  # Gibbs采样步数
                 learning_rate=1e-3,  # 学习率
                 momentum_coefficient=0.9,  # 动量系数
                 weight_decay=1e-4,  # 权重衰减
                 use_gpu=False,
                 _activation='sigmoid'):
        super(DBN, self).__init__()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        self.n_layers = len(hidden_units)  # 隐含层数
        self.rbm_layers = []  # rbm
        self.rbm_nodes = []
        # 构建不同的RBM层
        for i in range(self.n_layers):

            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_units[i - 1]
            rbm = RBM(visible_units=input_size,
                      hidden_units=hidden_units[i],
                      k=k,
                      learning_rate=learning_rate,
                      momentum_coefficient=momentum_coefficient,
                      weight_decay=weight_decay,
                      use_gpu=use_gpu,
                      _activation=_activation).to(device)

            self.rbm_layers.append(rbm)
        self.W_rec = [nn.Parameter(self.rbm_layers[i].weight.data.clone()) for i in range(self.n_layers - 1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].weight.data) for i in range(self.n_layers - 1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].c.data.clone()) for i in range(self.n_layers - 1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].b.data) for i in range(self.n_layers - 1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].weight.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].b.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].c.data)
        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])

        self.BPNN=nn.Sequential(            #用作分类和反向微调参数
            torch.nn.Linear(11, 11),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(11,5),
        )
    def forward(self , input_data):
        '''
            前馈
        '''
        v = input_data
        for i in range(len(self.rbm_layers)):
            v = v.view((v.shape[0] , -1)).type(torch.FloatTensor)#flatten
            p_v,v = self.rbm_layers[i].forward(v)
        # print('p_v:', p_v.shape,p_v)
        # print('v:',v.shape,v)
        out=self.BPNN(p_v)
        # print('out',out.shape,out)
        # print(self.BPNN(p_v))
        return out

    def train_static(self, train_data,train_labels,num_epochs,batch_size):
        '''
        逐层贪婪训练RBM,固定上一层
        '''

        tmp = train_data

        for i in range(len(self.rbm_layers)):
            print("-"*20)
            print("Training the {} st rbm layer".format(i+1))

            tensor_x = tmp.type(torch.FloatTensor)
            tensor_y = train_labels.type(torch.FloatTensor)
            _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
            _dataloader = torch.utils.data.DataLoader(_dataset)

            self.rbm_layers[i].trains(_dataloader,num_epochs,batch_size)
            print(type(_dataloader))
            # print(train_data.shape)
            v = tmp.view((tmp.shape[0] , -1)).type(torch.FloatTensor)
            v,_ = self.rbm_layers[i].forward(v)
            tmp = v
            # print(v.shape)
        return

    def train_ith(self, train_data,num_epochs,batch_size,ith_layer,rbm_layers):
        '''
        只训练某一层，可用作调优
        '''
        if(ith_layer>len(rbm_layers)):
            return

        v = train_data
        for ith in range(ith_layer):
            v,out_ = self.rbm_layers[ith].forward(v)


        self.rbm_layers[ith_layer].trains(v, num_epochs,batch_size)
        return

    def trainBP(self,trainloader):
        optimizer = torch.optim.SGD(self.BPNN.parameters(), lr=0.005, momentum=0.7)
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(5):
            for step,(x,y) in enumerate(trainloader):
                bx = Variable(x)
                by = Variable(y)
                out=self.forward(bx)[1]
                # print(out)
                loss=loss_func(out,by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print('Epoch: ', epoch, 'step:', step, '| train loss: %.4f' % loss.data.numpy())




