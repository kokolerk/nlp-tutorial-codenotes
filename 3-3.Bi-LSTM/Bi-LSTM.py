# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    words = sentence.split()    #单词组合
    for i, word in enumerate(words[:-1]):
        #前i个单词的标号
        input = [word_dict[n] for n in words[:(i + 1)]]
        #padding，0补齐长度为max_len
        input = input + [0] * (max_len - len(input))
        #target是第i+1单词
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        #bidirectonal=true，表示双向lstm
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Linear(n_hidden * 2, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        #除了hidden维度+1，剩下的同lstm模型
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1*2, len(X), n_hidden)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), n_hidden)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    n_hidden = 5 # number of hidden units in one cell

    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )
    #语料库
    word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
    number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
    n_class = len(word_dict)
    max_len = len(sentence.split())
    #模型定义
    model = BiLSTM()
    #损失函数，优化方式定义
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #数据预处理
    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    #once again，梯度归零，前向传播，计算loss，反向传播，参数优化
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()
    #测试
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict.squeeze()])
'''
测试效果不太好，长句子后面预测好，前面预测的反而不是很准确……
'''