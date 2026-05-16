import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import random

#超参数
EMBEDDING_DIM = 50 #词向量维度，要和word2vec中词向量维数相同
MAX_LEN = 100
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-4
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

#为numpy,Dropout,pytorch设定固定seed,方便复现
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   


#数据处理
def load_data(file_path):
    #标签和待处理文本
    labels, texts = [], []
    with open(file_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            labels.append(int(parts[0]))
            texts.append(parts[1:])
        return labels, texts

#下面是一系列用于数据增强的函数
#随机删除
def random_deletion(words, p=0.1):
    if len(words) <= 1:
        return words
    new_words = [w for w in words if random.random() > p]
    return new_words if len(new_words) > 0 else [random.choice(words)]

#随机交换两个词
def random_swap(words, n=1):
    new_words = words[:]
    if len(new_words) < 2:
        return new_words
    for _ in range(n):
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words

#同义词替换词表
SYNONYM_DICT = {
    "好": ["不错", "优秀", "挺好"],
    "喜欢": ["喜爱", "钟爱"],
    "满意": ["满足", "认可"],
    "差": ["不好", "糟糕"],
    "失望": ["难过", "遗憾"],
    "垃圾": ["很差", "糟糕"]
}

#同义词替换
def synonym_replacement(words, replace_prob=0.1):
    new_words = []
    for word in words:
        if word in SYNONYM_DICT and random.random() < replace_prob:
            new_words.append(random.choice(SYNONYM_DICT[word]))
        else:
            new_words.append(word)
    return new_words

#数据增强文本
def augment_text(words):
    aug_words = words[:]

    op = random.choice(["delete", "swap", "synonym", "none"])

    if op == "delete":
        aug_words = random_deletion(aug_words, p=0.1)
    elif op == "swap":
        aug_words = random_swap(aug_words, n=1)
    elif op == "synonym":
        aug_words = synonym_replacement(aug_words, replace_prob=0.15)

    return aug_words

#增强训练数据集
def augment_training_data(labels, texts, aug_times=1):
    new_labels = labels[:]
    new_texts = texts[:]

    for label, text in zip(labels, texts):
        for _ in range(aug_times):
            aug_text = augment_text(text)
            if aug_text != text:
                new_labels.append(label)
                new_texts.append(aug_text)

    return new_labels, new_texts

#构造Dataset
class TextDataset(Dataset):
    def __init__(self, labels, texts, word2idx, max_len):
        self.labels = torch.tensor(labels, dtype = torch.float32)
        #以id形式存储文本
        self.texts = []
        #文本的长度
        self.lengths = []
        for text in texts:
            #将词转换为索引，未登录词使用<UNK>的索引
            idx_seq = [word2idx.get(w,1) for w in text]
            seq_len = max(1, min(len(idx_seq), max_len))
            #截断或填充
            if len(idx_seq) < max_len:
                idx_seq += [0]*(max_len - len(idx_seq))
            else:
                idx_seq = idx_seq[:max_len]
            self.texts.append(idx_seq)
            self.lengths.append(seq_len)
        self.texts = torch.tensor(self.texts, dtype = torch.long)
        self.lengths = torch.tensor(self.lengths, dtype = torch.long)
    
    #返回文本长度
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.lengths[idx]

#返回word2idx和嵌入矩阵
def build_vocab_and_embeddings(w2v_path, texts_train):
    print("正在加载预训练词向量")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary = True)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    embeddings = [np.zeros(EMBEDDING_DIM), np.random.normal(scale = 0.1, size=(EMBEDDING_DIM,))]
    
    idx = 2
    for text in texts_train:
        for word in text:
            if word not in word2idx:
                word2idx[word] = idx
                if word in w2v_model:
                    embeddings.append(w2v_model[word])
                else:
                    embeddings.append(np.random.normal(scale = 0.1,size=(EMBEDDING_DIM,)))
                idx+=1
    embeddings = np.array(embeddings, dtype = np.float32)
    return word2idx, torch.tensor(embeddings)

#全连接神经网络
class MLP(nn.Module):
    def __init__(self, embedding_matrix):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False, padding_idx=0)
        self.fc1 = nn.Linear(EMBEDDING_DIM, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,1)
    
    def forward(self, x, lengths = None):
        # x shape: (batch, max_len)
        if lengths is None:
            lengths = x.ne(0).sum(dim = 1).clamp(min = 1)
        mask = x.ne(0).unsqueeze(-1).float()
        x = self.embedding(x) #(batch, max_len, embed_dim)
        x = (x * mask).sum(dim = 1) / lengths.unsqueeze(1).float()
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x).squeeze(1)
        return x
    
#卷积神经网络
class TextCNN(nn.Module):
    def __init__(self, embedding_matrix):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False, padding_idx = 0)
        self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=100, kernel_size = k) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * 3, 1)
    def forward(self, x, lengths = None):
        #x :(batch, max_len)
        if lengths is None:
            lengths = x.ne(0).sum(dim = 1).clamp(min = 1)
        x = self.embedding(x)#(batch, max_len, embed_dim)
        x = x.permute(0,2,1)
        conv_outs = []
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
            c = torch.relu(conv(x))#(batch, out_channels, L_out)
            valid_lengths = (lengths - kernel_size + 1).clamp(min = 1)
            positions = torch.arange(c.size(2), device = c.device).unsqueeze(0)
            conv_mask = positions < valid_lengths.unsqueeze(1)
            c = c.masked_fill(~conv_mask.unsqueeze(1), torch.finfo(c.dtype).min)
            p = torch.max(c, dim = 2)[0] #(batch, out_channels)
            conv_outs.append(p)
        
        x = torch.cat(conv_outs, dim=1)#(batch, 300)
        x = self.dropout(x)
        x = self.fc(x).squeeze(1)
        return x


#LSTM
class TextLSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(input_size = EMBEDDING_DIM, hidden_size = 128, num_layers = 1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128*2, 1)
    
    def forward(self, x, lengths = None):
        #x shape:(batch, max_len)
        if lengths is None:
            lengths = x.ne(0).sum(dim = 1).clamp(min = 1)
        x = self.embedding(x) #(batch, max_len, embde_dim)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first = True, enforce_sorted = False)
        _, (h_n, c_n) = self.lstm(packed_x)
        hidden = torch.cat((h_n[-2], h_n[-1]), dim = 1)#(batch, hidden*2)
        x = self.dropout(hidden)
        x = self.fc(x).squeeze(1)
        return x

#评测函数
def evaluate_model(model, dataloader, criterion=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts, labels, lengths = texts.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (outputs > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            true_positive += ((preds == 1) & (labels == 1)).sum().item()
            false_positive += ((preds == 1) & (labels == 0)).sum().item()
            false_negative += ((preds == 0) & (labels == 1)).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return correct / total, total_loss / len(dataloader), precision, recall, f1_score

#训练函数，加入早停机制
def train_model(model, train_loader, val_loader, model_name = "Model"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    model.to(DEVICE)
    best_val_loss = float("inf")
    best_state_dict = None
    #连续loss未下降次数
    patience_counter = 0
    print(f"\n-------------开始训练{model_name}-----------------")
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        for texts, labels, lengths in train_loader:
            texts, labels, lengths = texts.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            #线性层输出>0为1，<0为0
            preds = (outputs>0).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        train_acc = correct_train / total_train
        val_acc, val_loss, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion)
        print(f"""Epoch[{epoch+1}/{MAX_EPOCHS}] | Train Loss:
              {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} |Val Loss:{val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}""")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"验证集损失连续 {patience_counter} 轮未下降")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"{model_name} 触发早停，在第 {epoch + 1} 轮结束训练")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"{model_name} 已恢复到验证集损失最低时的模型参数")

def main():
    #固定seed
    seed_everything(SEED)
    #加载文本数据
    train_labels, train_texts = load_data('Dataset/train.txt')
    val_labels, val_texts = load_data('Dataset/validation.txt')
    test_labels, test_texts = load_data('Dataset/test.txt') 
    #数据增强
    #train_labels, train_texts = augment_training_data(train_labels,train_texts,aug_times=1)
    #加载word2idx和嵌入矩阵
    word2idx, embedding_matrix = build_vocab_and_embeddings('Dataset/wiki_word2vec_50.bin', train_texts)
    train_dataset = TextDataset(train_labels, train_texts, word2idx, MAX_LEN)
    val_dataset = TextDataset(val_labels, val_texts,word2idx, MAX_LEN )
    test_dataset = TextDataset(test_labels, test_texts, word2idx, MAX_LEN)
    #构造三者DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE, shuffle = False)
    models = {
        "MLP": MLP(embedding_matrix),
        "TextCNN": TextCNN(embedding_matrix),
        "TextLSTM": TextLSTM(embedding_matrix)
    }
    best_model_name = None
    best_model_f1 = -1.0
    for name, model in models.items():
        train_model(model, train_loader, val_loader, model_name=name)
        test_acc, test_loss, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader)
        print(f"{name}在测试集上的准确率:{test_acc:.4f}")
        print(f"{name}在测试集上的Precision:{test_precision:.4f} | Recall:{test_recall:.4f} | F1-score:{test_f1:.4f}")

        if test_f1 > best_model_f1:
            best_model_f1 = test_f1
            best_model_name = name

    target_model = models[best_model_name]

    print("\n" + "="*40)
    print(f"--- 情感分析实时推理 ({best_model_name}) ---")
    print("输入 'q' 退出程序")
    
    while True:
        user_input = input("\n请输入一段中文评论: ")
        if user_input.lower() == 'q':
            break
        
        if not user_input.strip():
            continue
            
        result, score = predict_sentiment_chinese(user_input, target_model, word2idx)
        
        print(f"预测判定: {result}")
        print(f"正面概率: {score:.4f}")
    print("程序已退出。")
import jieba

def predict_sentiment_chinese(text, model, word2idx, max_len=MAX_LEN):
    """
    针对中文文本的预测函数
    text: 输入的原始中文字符串
    model: 训练好的模型
    word2idx: 训练时生成的词汇索引表
    """
    model.eval()  # 切换到评估模式，关闭 Dropout
    
    # 1. 使用 jieba 进行中文分词
    words = jieba.lcut(text.strip())
    print(f"分词结果: {' / '.join(words)}") 

    # 2. 将词汇转换为索引
    # 0 是 <PAD>, 1 是 <UNK>
    idx_seq = [word2idx.get(w, 1) for w in words]
    
    # 3. 填充或截断 (与 TextDataset 中的逻辑一致)
    seq_len = max(1, min(len(idx_seq), max_len))
    if len(idx_seq) < max_len:
        idx_seq += [0] * (max_len - len(idx_seq))
    else:
        idx_seq = idx_seq[:max_len]
        
    # 4. 转换为 Tensor 并增加 Batch 维度 [1, max_len]
    input_tensor = torch.tensor([idx_seq], dtype=torch.long).to(DEVICE)
    length_tensor = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        # 5. 模型推理
        output = model(input_tensor, length_tensor)
        # 6. 计算概率
        # 模型最后是线性层，需要用 sigmoid 转换为 0-1 之间的概率
        probability = torch.sigmoid(output).item()
        
    # 7. 判定分类
    if probability < 0.5:
        return "正面 (Positive) 😊", probability
    else:
        return "负面 (Negative) 😡", probability
if __name__ == "__main__":
    main()
