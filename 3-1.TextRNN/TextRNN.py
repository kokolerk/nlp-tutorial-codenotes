# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#功能和NNLM一样，输入前n-1个词，预测第n个词
def make_batch():
    '''
    预处理数据
    input_batch为前n-1个词的one-hot向量
    output_batch为第n个词的标号
    '''
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'
        #这个设置one-hot的方法很简便
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        #直接调用pytorch里面的RNN模型
        #input_size是输入向量的维度，hidden_size是隐藏层的维度
        #n_class = len(word_dict)为token的个数，7
        #n_hidden这里是5
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    n_step = 2 # number of cells(= number of Step)
    n_hidden = 5 # number of hidden units in one cell

    sentences = ["i like dog", "i love coffee", "i hate milk"]
    #语料库
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    #[3,2,7]3组数据，每组2个token，每个token的one-hot向量长度为7
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)#tensor([5, 4, 2])

    # Training
    for epoch in range(2000):
        optimizer.zero_grad()

        # hidden : [num_layers * num_directions, batch, hidden_size]
        #初始化h0，默认都是0
        hidden = torch.zeros(1, batch_size, n_hidden)
        # input_batch : [batch_size, n_step, n_class]
        output = model(hidden, input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()
    #又是用训练集当验证集
    input = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])