import numpy as np
import pandas as pd
import codecs
import matplotlib
import matplotlib.pyplot as plt
import random
import codecs
# this input_path is just example
#inputpath = "C:/Users/Administrator/Desktop/Smote/describledata/kddcup.data_10_percent_corrected"
Nor_list = ['normal']
Dos_list = ['back', 'land', 'neptune' , 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']
R2L_list =['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'xlock','xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel','spy','named','snmpguess']
Probe_list = ['satan','ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
U2R_list =['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps','httptunnel']
continue_col_index = [0,4,5,7,8,9,10,12,15,16,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
decreate_col_index =[1,2,3,6,11,13,14,20,21]


def gen_lable1(seg):
    if seg in Nor_list:
        return 0
    elif seg in Dos_list:
        return 1
    elif seg in R2L_list:
        return 2
    elif seg in Probe_list:
        return 3
    elif seg in U2R_list:
        return 4


def gen_label2(seg):
    if seg in Nor_list:
        return 0
    else:
        return 1


def replace(columnlist, data):
    # 离散值代替
    # listt = []
    for i in columnlist:
        listt = []
        for line, seg in enumerate(data[:, i]):
            if seg in listt:
                data[line][i] = listt.index(seg)
            else:
                listt.append(seg)
                data[line][i] = listt.index(seg)


def processdata(file_path):
    tmp = None
    with codecs.open(file_path, 'r') as f:
        content = f.readlines()
        datas = []
        for line in content:
            line = line.strip()
            line = line[:-1].split(',')
            datas.append(line)
        datas = np.array(datas)
        replace(decreate_col_index, datas)
        new_datas = []
        for index, col in enumerate(datas):
            new_datas.append([float(k) for k in col[:-1]])
            if random.random() < 0.00005:
                print(col[-1])
            new_datas[index].append(gen_lable1(col[-1]))
            new_datas[index].append(gen_lable1(col[-1]))
        tmp = new_datas
    datas = np.array(tmp).astype('float32')

    for j in continue_col_index:
        meanVal = np.mean(datas[:, j])
        stdVal = np.std(datas[:, j])
        datas[:, j] = (datas[:, j] - meanVal) / stdVal
    return datas
def gen_train_test(dat):        #分割训练集和测试集
    index = int(len(dat)*0.8)
    np.random.shuffle(dat)
    traindat=torch.from_numpy(dat[:index,:-2]).float()
    trainlabel=torch.from_numpy(dat[:index,-2]).long()
    validdat = torch.from_numpy(dat[index:,:-2]).float()
    validlabel = torch.from_numpy(dat[index:,-2]).long()
    # print(dat[0])
    return traindat,trainlabel,validdat,validlabel

def train_batch(traind,trainl,SIZE=200,SHUFFLE=True):   #分批处理
    trainset=Data.TensorDataset(traind,trainl)
    trainloader=Data.DataLoader(
        dataset=trainset,
        batch_size=SIZE,
        shuffle=SHUFFLE)
    return trainloader

def train_and_test(traind,trainl,testdat,testlabel,loader):
    print(type(traind),type(trainl),type(traind),type(trainl),type(loader))
    start_time = time.time()
    dbn=DBN(visible_units=len(traind[0]))

    dbn.train()
    dbn.train_static(train_data=traind,train_labels=trainl,num_epochs=0,batch_size=20)

    optimizer = torch.optim.SGD(dbn.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loader = loader
    dbn.trainBP(train_loader)

    for epoch in range(0):
        for step,(x,y) in enumerate(train_loader):
            # print(x.data.numpy(),y.data.numpy())

            b_x=Variable(x)
            b_y=Variable(y)

            output=dbn(b_x)
            # print(output)
            # print(prediction);print(output);print(b_y)

            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%10==0:
                print('Epoch: ', epoch, 'step:',step,'| train loss: %.4f' % loss.data.numpy())
    duration=time.time()-start_time

    dbn.eval()
    test_x = Variable(testdat);test_y = Variable(testlabel)
    test_out = dbn(test_x)
    # print(test_out)
    test_pred = torch.max(test_out, 1)[1]
    pre_val = test_pred.data.squeeze().numpy()
    y_val = test_y.data.squeeze().numpy()
    print('prediciton:',pre_val);print('true value:',y_val)
    accuracy = float((pre_val == y_val).astype(int).sum()) / float(test_y.size(0))
    print('test accuracy: %.2f' % accuracy,'duration:%.4f' % duration)
    return accuracy, duration

















