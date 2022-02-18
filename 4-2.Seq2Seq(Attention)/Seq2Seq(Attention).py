# %%
# code by Tae Hwan Jung @graykode
# Reference : https://github.com/hunkim/PyTorchZeroToAll/blob/master/14_2_seq2seq_att.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#数据处理的方式和4-1一样
#也是一个生成任务，不过做的是机器翻译，从德文翻译为英文
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch():
    #这里面的batch就是一个，就一句话
    #前两个是one-hot向量，taget是标号
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]

    # make tensor
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)
#传说中的Attention机制终于来了
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)

        # Linear for attention
        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden * 2, n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1)  # enc_inputs: [n_step(=n_step, time step), batch_size, n_class]
        dec_inputs = dec_inputs.transpose(0, 1)  # dec_inputs: [n_step(=n_step, time step), batch_size, n_class]

        # enc_outputs : [n_step=5, batch_size=1, num_directions(=1) * n_hidden], matrix F
        # enc_hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden=128]
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)
        #之前的和4-1一样，都是编码
        trained_attn = []
        hidden = enc_hidden#编码最后输出的隐藏层，在4-1里面就是输入的句向量了……
        n_step = len(dec_inputs) #这里也是5，输出的词的个数
        model = torch.empty([n_step, 1, n_class])
        #对每个词，用attention机制
        for i in range(n_step):  # each time step
            # dec_output : [n_step(=1), batch_size(=1), num_directions(=1) * n_hidden]
            # hidden : [num_layers(=1) * num_directions(=1), batch_size(=1), n_hidden]
            # decoder部分的RNN和4-1差不多，
            # 不同的是，这里只把dec_inputs[i]作为输入，也就是第i个单词
            # 4-1里面是全部作为输入
            # ？？？不知道是不是理解的有问题，这里没有考虑y(i-1)对yi的影响……
            # 只考虑了encoder和yi对模型参数的影响
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            # 计算attention的值，详细注解见下，也就是want和所有德语词的权重关系在这里了
            attn_weights = self.get_att_weight(dec_output, enc_outputs)  # attn_weights : [1, 1, n_step]
            #加入全句attention
            trained_attn.append(attn_weights.squeeze().data.numpy())

            # matrix-matrix product of matrices [1,1,n_step] x [1,n_step,n_hidden] = [1,1,n_hidden]
            #权重 x 对应的encoder向量，计算上下文的向量
            context = attn_weights.bmm(enc_outputs.transpose(0, 1))
            dec_output = dec_output.squeeze(0)  # dec_output : [batch_size(=1), num_directions(=1) * n_hidden]
            context = context.squeeze(1)  # [1, num_directions(=1) * n_hidden]
            #decoder向量和上下文的attention向量拼接起来，进行FC输出，得到类别向量
            #？？？1不知道是啥意思
            model[i] = self.out(torch.cat((dec_output, context), 1))

        # make model shape [n_step, n_class]
        return model.transpose(0, 1).squeeze(0), trained_attn
    #求出每一个解码向量对应的编码编码的attention，权重
    #比如输出的单词是want，也就是dec_output，enc_outputs是编码一句话的的所有输出
    def get_att_weight(self, dec_output, enc_outputs):  # get attention weight one 'dec_output' with 'enc_outputs'
        #编码长度，这里是5
        n_step = len(enc_outputs)
        #初始化权重为0
        attn_scores = torch.zeros(n_step)  # attn_scores : [n_step]
        #计算每个enc的向量对应的权重
        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])

        # Normalize scores to weights in range 0 to 1，正则化处理
        return F.softmax(attn_scores).view(1, 1, -1)
    #计算1v1的权重，比如want ，ich对应的RNN向量
    def get_att_score(self, dec_output, enc_output):  # enc_outputs [batch_size, num_directions(=1) * n_hidden]
        #128-128的线性层
        score = self.attn(enc_output)  # score : [batch_size, n_hidden]
        #每一个输入的向量要和当前生成词的向量相乘,得到权重，
        return torch.dot(dec_output.view(-1), score.view(-1))  # inner product make scalar value

if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 128 # number of hidden units in one cell
    #输入，P补齐；输出，S打头；标签，E结尾。长度都是5
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    #三句话连起来
    word_list = " ".join(sentences).split()
    #去除相同的词
    word_list = list(set(word_list))
    #i是数，w是词，key=词
    word_dict = {w: i for i, w in enumerate(word_list)}
    #i是数，w是词，key=数
    number_dict = {i: w for i, w in enumerate(word_list)}
    #词的大小，目测是11（加上SEP）
    n_class = len(word_dict)  # vocab list

    # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
    #初始化hidden，h0
    hidden = torch.zeros(1, 1, n_hidden)
    #调用模型，定义损失函数，优化方式，学习率
    model = Attention()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #预处理数据
    input_batch, output_batch, target_batch = make_batch()

    # Train，开始训练了！套路都是一样的
    #2000次的attention结果不好，换成5000次试一试
    for epoch in range(5000):
        optimizer.zero_grad()
        #output输出不变，还是对应的n_class维度的概率向量
        output, _ = model(input_batch, hidden, output_batch)

        loss = criterion(output, target_batch.squeeze(0))
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    #这个test_batch是什么用啊？顶替的是训练的时候output_batch的位置
    #大概是随便定义的一个空结果吧，即使是测试，也不能什么也不输入
    test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
    test_batch = torch.FloatTensor(test_batch)
    #分类结果（11维向量，对应的是概率，选取最大的那个就是词了），attention值（详细注释见模型）
    predict, trained_attn = model(input_batch, hidden, test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # Show Attention
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    #权值越小，颜色越深，
    # 其实训练效果不是很理想……，没有对角线的奇观，反而是P，S之流的权重很大
    ax.matshow(trained_attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()