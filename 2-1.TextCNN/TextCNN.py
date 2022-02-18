# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#情感分类模型
#输入一句话，判断这个是好的情感还是坏的情感
class TextCNN(nn.Module):
    def __init__(self):
        # embedding_size = 2  # embedding size
        # sequence_length = 3  # sequence length
        # num_classes = 2  # number of classes
        # filter_sizes = [2, 2, 2]  # n-gram windows
        # num_filters = 3  # number of filters
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        # 设置词嵌入向量的词的个数，词嵌入向量的维度，这里vocab_size = len(word_dict)，embedding_size=3
        self.W = nn.Embedding(vocab_size, embedding_size)
        #最后全连接层的权重
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        #最后全连接层的偏置
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        #卷积层设置
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        #词嵌入向量，增加一个维度
        embedded_chars = self.W(X) # [batch_size, sequence_length, sequence_length]
        embedded_chars = embedded_chars.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        #卷积和池化
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)
        #池化结构全链接分类
        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = self.Weight(h_pool_flat) + self.Bias # [batch_size, num_classes]
        return model

if __name__ == '__main__':
    embedding_size = 2 # embedding size
    sequence_length = 3 # sequence length
    num_classes = 2 # number of classes
    filter_sizes = [2, 2, 2] # n-gram windows
    num_filters = 3 # number of filters

    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.
    #组建语料库
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNN()
    #损失函数是交叉熵，优化方式是Adam，学习率是0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #输入输出处理，输入为一句话各个词对应的标号
    #输出为1或者0
    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels]) # To using Torch Softmax Loss function

    # Training
    for epoch in range(5000):
        #梯度归0，前向传播
        optimizer.zero_grad()
        output = model(inputs)
        #计算loss
        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        #反向传播，参数优化
        loss.backward()
        optimizer.step()

    # Test
    test_text = 'sorry love you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text,"is Bad Mean...")
    else:
        print(test_text,"is Good Mean!!")