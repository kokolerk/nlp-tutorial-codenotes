# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#不全单词的最后一个字母
def make_batch():
    '''
    一个单词前几个字母为input
    最后一个字母是output
    input_batch:one-hot向量的组合
    output_batch:标号
    '''
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        #lstm函数已经被封装好了，input_size是输入向量的维度，hidden_size是隐藏层的维度
        #lstm+线性层输出
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]
        #有两个需要初始化，h0和c0
        hidden_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(X), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    #前n-1个字母，这里是前3个字母
    n_step = 3 # number of cells(= number of Step)
    #hidden的维度，也是cell的维度
    n_hidden = 128 # number of hidden units in one cell
    #字母库
    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n: i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    #这里是24
    n_class = len(word_dict)  # number of class(=number of vocab)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM()
    #交叉熵是损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(1000):
        #梯度归0
        optimizer.zero_grad()
        #前向传播，计算loss，反向传播，更新参数
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()
    #测试
    inputs = [sen[:3] for sen in seq_data]

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])