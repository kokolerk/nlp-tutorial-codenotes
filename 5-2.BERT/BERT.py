# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert
import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# sample IsNext and NotNext to be same in small batch size
def make_batch():
    """处理训练集, 获得模型输入 batch:[batch_size, 5]"""
    batch = []      #初始化batch列表
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2: #如果正负例没有达到batch_size的一半, 继续采样
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index] #随机采样两句话
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']] #输入id中加入分割符
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)    #构造段序号, 第一句话seg_id为0, 第二句话sed_id为1

        # MASK LM
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence, 随机选择15%的token
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]   #获得除特殊token外的index
        shuffle(cand_maked_pos)             #打乱顺序
        masked_tokens, masked_pos = [], []  #掩码token列表, 掩码index列表
        for pos in cand_maked_pos[:n_pred]: #选取15%的token处理
            masked_pos.append(pos)          #加入到掩码index列表
            masked_tokens.append(input_ids[pos])    #加入到掩码token列表
            if random() < 0.8:              #80%的情况下,用[MASK]标记替换单词
                input_ids[pos] = word_dict['[MASK]'] # make mask
            elif random() < 0.5:            #10%的情况下,用一个随机的替换该单词
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]] # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)     #需要pad的位数
        input_ids.extend([0] * n_pad)       #对输入词id进行padding
        segment_ids.extend([0] * n_pad)     #对输入段id进行padding

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:                   #若选出的词数未达到max_pred, 用0(表示[PAD])进行padding
            n_pad = max_pred - n_pred           #padding数量
            masked_tokens.extend([0] * n_pad)   #对token掩码列表进行padding
            masked_pos.extend([0] * n_pad)      #对index掩码列表进行padding

        # input_ids: [maxlen, ]
        # segment_ids: [maxlen, ]
        # masked_tokens: [max_pred, ]
        # masked_pos: [max_pred, ]

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:    #如果构成后继关系
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1   #正例数量加1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:  #如果不构成后继关系
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
    negative += 1   #负例数量加1

    return batch    #[batch_size, 5]
# Proprecessing Finished

def get_attn_pad_mask(seq_q, seq_k):
    """
    encoder注意力掩码矩阵，PAD token设置为1，其余为0，与transformer代码相同
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def gelu(x):
    """
    gelu激活函数
    """
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    """
    嵌入层
    """
    def __init__(self):
        super(Embedding, self).__init__()
        #三个embedding向量
        #词编码
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        #位置编码
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        #端编码
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        #正则化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        """
        词嵌入前向传播
        @param x:待Embedding的词序号序列 [batch_size, max_len]
               seq:待Embedding的段序号序列 [batch_size, max_len]
        @return 嵌入结果 [batch_size, max_len, d_model]
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    """
    同transformer.py注释,计算注意力权重，分数
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        公式为: Softmax(Q * K.T/ sqrt(d_k)) * V
        @param Q: [batch_size, n_heads, len_q, d_k]
               K: [batch_size, n_heads, len_q, d_k]
               V: [batch_size, n_heads, len_q, d_v]
               attn_mask: [batch_size, n_heads, len_q, len_q]
        @return context: 最终求得的注意力分数 [batch_size, n_heads, len_q, d_v]
                attn: 自注意力权重分布矩阵 [batch_size, n_heads, len_q, len_q]
        """
        # Q * K.T/ sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        # Softmax(Q * K.T/ sqrt(d_k))
        attn = nn.Softmax(dim=-1)(scores)#softmax归一化处理
        # Softmax(Q * K.T/ sqrt(d_k))* V
        context = torch.matmul(attn, V)#注意力权重求和
        return context, attn

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制，同transfomer.py
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        """
        多头注意力层前向传播
        @param Q: [batch_size, src_len, d_model]
               K: [batch_size, src_len, d_model]
               V: [batch_size, src_len, d_model]
               attn_mask: [batch_size, len_q, len_k]
        @return 多头注意力层输出结果 shape: [batch_size, len_q, d_model]
                attn: 注意力权重分布矩阵 shape: [batch_size, n_heads, len_q, len_k]
        """
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        #纬度填充，第二个纬度（注意力的头数上）进行重复扩张
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    """
    前向计算：两个线性层
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        前馈网络层的计算
        @param inputs:输入 shape: [batch_size, seq_len, d_model]
        @return 输出 shape: [batch_size, seq_len, d_model]
        """
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    """
    一层编码网络
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        #多头注意力模型
        self.enc_self_attn = MultiHeadAttention()
        #前向计算模型
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        单层编码器前向传播
        @param enc_inputs: 编码器输入 shape:[batch_size, src_len, d_model]
               enc_self_attn_mask: 编码器自注意力掩码矩阵 shape:[batch_size, len_q, len_k]
        @return enc_outputs: 编码器输出, shape:[batch_size, src_len, d_model]
                attn: 编码器自注意力权重分布矩阵, shape:[batch_size, n_heads, len_q, len_q]
        """
        #注意力层
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        #前馈网络层
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class BERT(nn.Module):
    '''
    bert模型
    '''
    def __init__(self):
        super(BERT, self).__init__() #初始化，继承父类方法
        self.embedding = Embedding() #输入，词嵌入向量
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])#n_layers层编码
        self.fc = nn.Linear(d_model, d_model) #线性层
        self.activ1 = nn.Tanh() #激活函数1 tanh
        self.linear = nn.Linear(d_model, d_model) #线性层
        self.activ2 = gelu #激活函数2 gelu
        self.norm = nn.LayerNorm(d_model) #正则化
        self.classifier = nn.Linear(d_model, 2) #分类器（正父类）
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight #获得token_embedding权重
        n_vocab, n_dim = embed_weight.size() #词表大小，embedding纬度
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False) #解码器，一个线性层
        self.decoder.weight = embed_weight #与embedding共享权重
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab)) #bias偏置

    def forward(self, input_ids, segment_ids, masked_pos):
        """
        bert前向传播
        @param input_ids:输入词id序列 shape:[batch_size, max_len]
               segment_ids:输入段id序列 shape:[batch_size, max_len]
               masked_pos:掩码位置index序列 shape:[batch_size, max_pred]
        @return logits_lm: Masked LM 输出结果 shape:[batch_size, max_pred, n_vocab]
                logits_clsf: 句子分类输出结果  shape: [batch_size, 2]
        """
        output = self.embedding(input_ids, segment_ids)  # 词嵌入向量编码
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # 获得自注意力掩码矩阵
        for layer in self.layers:  # 遍历每一层Encoder
            output, enc_self_attn = layer(output, enc_self_attn_mask)  # Encoder编码
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, len, len]
        # it will be decided by first token(CLS), 对第一个token对应输出进行变换
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2], 得到分类结果分数

        # 求出被mask位置的词，准备与正确词进行比较
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))  # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))  # mask后的矩阵经过线性层+激活函数+层归一化
        logits_lm = self.decoder(h_masked) + self.decoder_bias  # 解码操作 [batch_size, max_pred, n_vocab]

    return logits_lm, logits_clsf  # 返回模型输出结果


if __name__ == '__main__':
    # BERT Parameters
    maxlen = 30 # maximum of length
    batch_size = 6
    max_pred = 5  # max tokens of prediction
    n_layers = 6 # number of Encoder of Encoder Layer
    n_heads = 12 # number of heads in Multi-Head Attention
    d_model = 768 # Embedding Size
    d_ff = 768 * 4  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
    #语料
    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    #获得句子列表，过滤停用词
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
    #获得过滤后的词表，无重复
    word_list = list(set(" ".join(sentences).split()))
    #特殊token词典映射
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    #词表中的词的id映射
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    #id到词的映射词典
    number_dict = {i: w for i, w in enumerate(word_dict)}
    #词表大小
    vocab_size = len(word_dict)

    token_list = list() #输入的token列表
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()] #获取句子中每个token对应的id
        token_list.append(arr) #加入到toekn列表里面

    model = BERT()#定义模型
    criterion = nn.CrossEntropyLoss()#定义损失函数，交叉熵
    optimizer = optim.Adam(model.parameters(), lr=0.001) #参数优化

    batch = make_batch() #获得模型输入
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch)) #类型转换，转换为longtensor

    #训练
    for epoch in range(100):
        optimizer.zero_grad() #梯度归零
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos) #bert模型前向传播
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM，计算损失函数
        loss_lm = (loss_lm.float()).mean() #计算平均损失
        loss_clsf = criterion(logits_clsf, isNext) # for sentence classification，还是计算损失
        loss = loss_lm + loss_clsf #总的损失函数
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward() #反向传播，参数求导
        optimizer.step() #更新参数

    # Predict mask tokens ans isNext
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
    print(text) #打印text
    #打印测试集中的词
    print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])

    #bert前向传播计算
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    #获得logits_lm最大值对应的index（vocab，index）
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    # 打印真值结果(忽略PAD位)
    print('masked tokens list : ',[pos.item() for pos in masked_tokens[0] if pos.item() != 0])
    #打印预测结果
    print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

    # 获得logits_clsf最大值对应的index
    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    #打印真值结果
    print('isNext : ', True if isNext else False)
    #打印预测结果
    print('predict isNext : ',True if logits_clsf else False)
