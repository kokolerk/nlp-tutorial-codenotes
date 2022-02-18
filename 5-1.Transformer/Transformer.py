# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch(sentences):
    '''
    预处理encoder输入,decoder输入，decoder正确输出
    语料中的token转化为字典中的索引值，进一步转化为longtensor
    tensor([[1, 2, 3, 4, 0]]) tensor([[5, 1, 2, 3, 4]]) tensor([[1, 2, 3, 4, 6]])
    @return input_batch:  encoder的输入(字符序列对应id) shape:[batch_num, n_step]
            output_batch: decoder的输入(字符序列对应id) shape:[batch_num, n_step]
            target_batch: decoder的输出真值(字符序列对应id), shape:[batch_num, n_step]
    '''
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    #print(torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch))
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position, d_model):
    '''
    计算位置向量，正弦位置编码
    @para:
    n_position
    d_model=512,词嵌入向量的纬度
    @return:
    返回编码结果, [n_position, d_model]
    '''
    def cal_angle(position, hid_idx): #计算位置对应角度
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position): #获得对应位置向量
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    # 计算每个位置的角度向量
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    # dim 2i，偶数用sin计算
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # dim 2i+1，奇数用cos计算
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    encoder注意力掩码矩阵，将PAD TOKEN的位置设置为1，其余为0
    '''
    #batch_size和Query的序列长度
    batch_size, len_q = seq_q.size()
    #batch_size和Key的序列长度
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    #unsqueeze(1)数据解压，在位置为1的纬度上扩充为1
    #eq（0），取
    # print('*******get_attn_pad_mask**********')
    # print('get_attn_pad_mask,seq_k.data.eq(0)',seq_k.data.eq(0))
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    #expand，合并？
    # print('pad_attn_mask.expand(batch_size, len_q, len_k)',pad_attn_mask.expand(batch_size, len_q, len_k))
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    '''
    decoder注意力掩码矩阵，将未解码的位置设置为1，其余为0
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):# 初始化，集成父类方法
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        scaled dot-product attention
        公式为: Softmax(Q * K.T/ sqrt(d_k)) * V
        @param Q: [batch_size, n_heads, len_q, d_k]
               K: [batch_size, n_heads, len_q, d_k]
               V: [batch_size, n_heads, len_q, d_v]
               attn_mask: [batch_size, n_heads, len_q, len_q]
        @return context: 注意力分数 [batch_size, n_heads, len_q, d_v]
                attn: 自注意力权重分布矩阵 [batch_size, n_heads, len_q, len_q]
        '''
        #矩阵乘法
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # Fills elements of self tensor with value where mask is one.
        # 掩码操作, 用value(=-1e9)填充tensor中与mask中值为1位置相对应的元素, 归一化之后对应权重趋近于0
        scores.masked_fill_(attn_mask, -1e9)
        #Softmax输出
        attn = nn.Softmax(dim=-1)(scores)
        #乘value
        context = torch.matmul(attn, V)
        #返回权值，attention值
        return context, attn

class MultiHeadAttention(nn.Module):
    '''
    多头注意力机制
    '''
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        #d_model=512,n_heads=8，8头
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        #正则层
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        '''
        在transformer里面，Q，K，V都是输入的向量，计算注意力权重
        多头注意力层前向传播
        @param Q: [batch_size, src_len, d_model]
               K: [batch_size, src_len, d_model]
               V: [batch_size, src_len, d_model]
               attn_mask: [batch_size, len_q, len_k]
        @return 多头注意力层输出结果 shape: [batch_size, len_q, d_model]
                attn: 注意力权重分布矩阵 shape: [batch_size, n_heads, len_q, len_k]
        '''
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        #线性层，纬度转化
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        #纬度扩展
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        # attention计算
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        #线性层
        output = self.linear(context)
        #正则化，output+input，避免经过multiattention之后性能下降
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    '''
    feedforward机制，两个卷积层+正则化
    '''
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        #卷积层，正则化
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        '''
        前馈网络层的计算 计算公式: LayerNorm(X + FeedForward(X))
       @param inputs:输入 shape: [batch_size, seq_len, d_model]
       @return 输出 shape: [batch_size, seq_len, d_model]
        '''
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    '''
    encoder一层网络模型
    一个多头注意力机制+卷积层实现，详情见论文的示意图
    '''
    def __init__(self):
        super(EncoderLayer, self).__init__()
        #两个：MultiHeadAttention+FeedForward
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        #前向计算
        '''
        单层解码器前向传播
        @param dec_inputs: decoder输入 shape: [batch_size, n_step, d_model]
               enc_outputs: encoder输出 shape:[batch_size, src_len, d_model]
               dec_self_attn_mask: 自注意力掩码矩阵 shape:[batch_size, tgt_len, tgt_len]
               dec_enc_attn_mask: Decoder-Encoder掩码矩阵 shape:[batch_size, tgt_len, src_len]
        @return dec_outputs: decoder输出 shape:[batch_size, tgt_len, d_model]
                dec_self_attn: decoder自注意力权重分布 shape:[batch_size, n_heads, tgt_len, tgt_len]
                dec_enc_attn: decoder-encoder注意力权重分布 shape:[batch_size, n_heads, tgt_len, src_len]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    '''
    decoder一层网络模型
    两个多头注意力机制+卷积层实现，详情见论文的示意图
    '''
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        单层解码器前向传播
        @param dec_inputs: decoder输入 shape: [batch_size, n_step, d_model]
               enc_outputs: encoder输出 shape:[batch_size, src_len, d_model]
               dec_self_attn_mask: 自注意力掩码矩阵 shape:[batch_size, tgt_len, tgt_len]
               dec_enc_attn_mask: Decoder-Encoder掩码矩阵 shape:[batch_size, tgt_len, src_len]
        @return dec_outputs: decoder输出 shape:[batch_size, tgt_len, d_model]
                dec_self_attn: decoder自注意力权重分布 shape:[batch_size, n_heads, tgt_len, tgt_len]
                dec_enc_attn: decoder-encoder注意力权重分布 shape:[batch_size, n_heads, tgt_len, src_len]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    '''
    编码模型
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        # src_vocab_size = len(src_vocab)#字典大小
        # d_model = 512  词嵌入向量纬度
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        #位置词向量
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        #层 n_layers = 6  # number of Encoder of Decoder Layer，与论文一致
        #6层子网
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        '''
        Encoder(6层)模型前向传播, 对输入编码
        @param enc_inputs:编码器输入 shape: [batch_size, src_len]
        @return enc_outputs:编码器输出 shape: [batch_size, src_len, d_model]
                enc_self_attns:编码器自注意力权重分布 shape: [n_layers, batch_size, n_heads, len_q, len_q]
        '''
        # enc_inputs=tensor([[1, 2, 3, 4, 0]]),为输入词向量
        print('enc_inputs_encoder',enc_inputs)
        #embedding + positional encoding
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        #
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    '''
    解码模型，与上文的encoder模型构成相似，代码逻辑相似
    '''
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        '''
         Decoder(6层)模型前向传播, 对输入进行解码
        @param dec_inputs:解码器输入 shape:[batch_size, n_step]
               enc_inputs:编码器输入 shape:[batch_size, n_step]
               enc_outputs:编码器输出 shape:[batch_size, src_len, d_model]
        @return dec_outputs:解码器输出 shape:[batch_size, tgt_len, d_model]
                dec_self_attns:解码器自注意力权重分布列表 shape:[n_layers, batch_size, n_heads, tgt_len, tgt_len]
                dec_enc_attns:解码器-编码器注意力权重分布列表 shape:[n_layers, batch_size, n_heads, tgt_len, src_len]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        #遍历decoder每一层
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder() #编码模型
        self.decoder = Decoder() #解码模型
        #针对翻译任务，连接线性层网络
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) #线性层，最后输出
    def forward(self, enc_inputs, dec_inputs):
        '''
        逻辑为：先编码，再解码，最后将输出的词嵌入向量转化为输出词向量，输出结果
        @param enc_inputs:编码器输入, shape:[batch_size, n_step]
               dec_inputs:解码器输入, shape:[batch_size, n_step]
        @return 模型输出, shape:[batch_size * tgt_len, tgt_vocab_size]
                enc_self_attns:编码器自注意力权重分布 shape:[n_layers, batch_size, n_heads, src_len, src_len]
                dec_self_attns:解码器自注意力权重分布 shape:[n_layers, batch_size, n_heads, tgt_len, tgt_len]
                dec_enc_attns: 解码器-编码器注意力权重分布 shape:[n_layers, batch_size, n_heads, tgt_len, src_len]
        '''
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def showgraph(attn):
    '''
    可视化注意力机制的权值
    然而可能是训练的语料库太少了，所以效果并不好，即时增大训练次数也是一样的
    没有出现对角线权值大的情况，反而是开头结尾比较大
    '''
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()

if __name__ == '__main__':
    # S: Symbol that shows starting of decoding input
    # E: Symbol that shows starting of decoding output
    # P: Symbol that will fill in blank sequence if current batch data size is short than time steps
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}#输入语料字典
    src_vocab_size = len(src_vocab)#字典大小

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}#输出语料字典
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}#key是数字标号，value是输出token
    tgt_vocab_size = len(tgt_vocab)#输出字典大小

    src_len = 5 # length of source
    tgt_len = 5 # length of target

    d_model = 512  # Embedding Size，论文有写，所有的输出均为512维
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    #transformer网络
    model = Transformer()
    #定义损失函数——交叉熵，定义优化方式adam，学习率为lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #规范化输入输出，形成batch
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    #训练
    for epoch in range(30):
        #梯度归零
        optimizer.zero_grad()
        #计算参数
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        #计算损失函数
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        #反向传播
        loss.backward()
        #参数更新
        optimizer.step()

    # Test
    #输出为每个单词的概率，选择概率最大的单词
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # Encoder最后一层的第一个Attention head的权重分布
    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    # Decoder最后一层自注意力的第一个head的权重分布
    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    # Decoder最后一层的Decoder-Encoder间第一个head的注意力权重分布
    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)