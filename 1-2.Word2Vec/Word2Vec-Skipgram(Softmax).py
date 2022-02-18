# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#数据随机采样
def random_batch():
    random_inputs = []
    random_labels = []
    #这里的batch_size是2，随机选取两个
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        #one-hot向量
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target
        #target标签
        random_labels.append(skip_grams[i][1])  # context word

    return random_inputs, random_labels

# Model
#生成词嵌入向量，用skip-Gram方法,本质上是用token预测他的上下文
#两个线性层
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # W and WT is not Traspose relationship
        # embedding_size这里是2
        self.W = nn.Linear(voc_size, embedding_size, bias=False) # voc_size > embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False) # embedding_size > voc_size Weight

    def forward(self, X):
        '''
        前向传播网络
        '''
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return output_layer

if __name__ == '__main__':
    batch_size = 2 # mini-batch size
    embedding_size = 2 # embedding size
    #语料库
    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]
    #没有去除相同token
    word_sequence = " ".join(sentences).split()
    #去除了相同token
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # Make skip gram of one size window
    skip_grams = []
    #为什么1,len-1，为了后面取前后两个token
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])
    #模型初始化
    model = Word2Vec()
    #定义损失函数，优化方法，学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training，过程类似
    for epoch in range(10000):
        #处理数据
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)
        #梯度归零
        optimizer.zero_grad()
        #前向传播，输入标签，输出他的上下文的词嵌入向量
        output = model(input_batch)
        #计算损失函数
        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        #反向传播+参数更新
        loss.backward()
        optimizer.step()
    #可视化词嵌入向量
    #目标：语义相近的靠的近，语义不同的靠的远
    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
