# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
#任务：生成模型，根据decoder-encoder模型，两个RNN实现
#具体的生成任务是生成反义词
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
#SEP各自的意义，字符表里面唯三的大写字母
def make_batch():
    input_batch, output_batch, target_batch = [], [], []
    # seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'],
    for seq in seq_data:
        #用P来长度补齐，n_step是5
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))
        #eg，input是man各字母的标号，
        # output是S+woman各字母的标号，
        # target是woman+E各字母标号
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]
        #转化为one-hot向量格式
        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        #target不是one-hot向量，是标号
        target_batch.append(target) # not one-hot

    # make tensor
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)

# make test batch，such as ‘man’
def make_testbatch(input_word):
    input_batch, output_batch = [], []
    #P补齐
    input_w = input_word + 'P' * (n_step - len(input_word))
    #转换为数字标号
    input = [num_dic[n] for n in input_w]
    #输出是SPPPPP
    output = [num_dic[n] for n in 'S' + 'P' * n_step]
    #转换为one-hot向量
    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]
    #make tensor
    return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)

# Model，主要的函数出现了！
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        #初始化RNN，两个，编码，解码
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        #FC全链接层，输出的是字符对应的数字
        self.fc = nn.Linear(n_hidden, n_class)
    #前向传播
    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1) # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = dec_input.transpose(0, 1) # dec_input: [max_len(=n_step, time step), batch_size, n_class]
        #编码，输出的是最后的ht
        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        #解码，encoder的输出ht为解码的h0
        outputs, _ = self.dec_cell(dec_input, enc_states)
        #全链接层，输出类别，格式已经标注了……
        model = self.fc(outputs) # model : [max_len+1(=6), batch_size, n_class]
        return model
# Test，一次训练一个单词，比如man
def translate(word):
    #转换格式
    input_batch, output_batch = make_testbatch(word)
    #运行训练好的模型来测试
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden)
    #这里的output_batch是SPPPPP，相当于没有什么用了……，就是占个位置
    output = model(input_batch, hidden, output_batch)
    # output : [max_len+1(=6), batch_size(=1), n_class]
    #输出编号，转换为词
    predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
    decoded = [char_arr[i] for i in predict]
    #去掉E
    end = decoded.index('E')
    translated = ''.join(decoded[:end])
    #去掉P（补齐的P）
    return translated.replace('P', '')

if __name__ == '__main__':
    n_step = 5
    n_hidden = 128
    #字符表
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    #i是数字，n是字符
    num_dic = {n: i for i, n in enumerate(char_arr)}
    #反义词组
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
    #词表大小
    n_class = len(num_dic)
    #数据多少
    batch_size = len(seq_data)
    #调用模型
    model = Seq2Seq()
    #损失函数是交叉熵
    criterion = nn.CrossEntropyLoss()
    #优化方式是Adam，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #初始化数据，分别为输入，输出，标签
    input_batch, output_batch, target_batch = make_batch()
    #开始训练
    for epoch in range(5000):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        #初始化h0
        hidden = torch.zeros(1, batch_size, n_hidden)
        #梯度归零
        optimizer.zero_grad()
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot
        #计算结果
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1, batch_size, n_class]
        # 格式转换
        output = output.transpose(0, 1) # [batch_size, max_len+1(=6), n_class]
        #计算loss值
        loss = 0
        for i in range(0, len(target_batch)):
            # output[i] : [max_len+1, n_class, target_batch[i] : max_len+1]
            #计算每一个词对的损失
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        #反向传播
        loss.backward()
        #参数优化
        optimizer.step()
# 测试
    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))