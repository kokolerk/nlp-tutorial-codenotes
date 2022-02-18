# %%
# code by Tae Hwan Jung @graykode
import torch
import torch.nn as nn
import torch.optim as optim
#神经网络语言模型
def make_batch():
    '''
    数据预处理模型，前n-1个单词存到input_batch里面
    第n个单词存到target_batch里面，作为输出标签
    '''
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split() # space tokenizer
        input = [word_dict[n] for n in word[:-1]] # create (1~n-1) as input
        target = word_dict[word[-1]] # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Model模型主函数
class NNLM(nn.Module):
    '''
    NNLM模型主体
    '''
    def __init__(self):
        super(NNLM, self).__init__()
        #嵌入词向量，输出维度为m
        self.C = nn.Embedding(n_class, m)
        #n_step是前多少个词,此处为2
        #n_hidden是隐藏层的神经元个数，这里是2
        #bias=False是无偏置项
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        #tanh里面的偏置，初始化为1
        self.d = nn.Parameter(torch.ones(n_hidden))
        #tanh结果输出后再进入线性层U，进入分类
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        #词向量进入线性层
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        #词向量进入线性层输出结果的偏置
        self.b = nn.Parameter(torch.ones(n_class))
    #前馈计算，一层线性层+一层tanh层
    def forward(self, X):
        """
        前向传播网络，后面所有的模型都会有这样一个前向传播网络
        """
        X = self.C(X) # X : [batch_size, n_step, m]
        X = X.view(-1, n_step * m) # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.H(X)) # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # [batch_size, n_class]
        return output

if __name__ == '__main__':
    n_step = 2 # number of steps, n-1 in paper
    n_hidden = 2 # number of hidden size, h in paper
    m = 2 # embedding size, m in paper
    #训练的数据集，根据前两个token预测第3个token
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    #制作词库,以空格为分隔符进行分割
    word_list = " ".join(sentences).split()
    #set——去除重复的token，list——转换格式为list
    word_list = list(set(word_list))
    #为单词制作字典，每个单词对应一个数字
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # number of Vocabulary
    #NNLM的model初始化
    model = NNLM()
    #损失函数是交叉熵
    criterion = nn.CrossEntropyLoss()
    #优化方式是Adam，学习率是0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #预处理数据，数据标签分开，转化为longTensor类型
    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training训练5000次
    for epoch in range(5000):
        #梯度归零
        optimizer.zero_grad()
        #前向传播
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        #计算损失
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        #反向传播，计算梯度
        loss.backward()
        #参数优化
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])