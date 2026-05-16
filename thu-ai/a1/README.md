# 拼音输入法

## 项目概述
本项目实现了一个基于统计语言模型的拼音输入法，旨在将拼音序列转换为最可能的汉字句子。项目包含二元及三元语言模型，并实现了 Viterbi 算法进行解码。同时提供了剪枝优化策略以提高长句处理效率。



## 运行方法
1. **基础运行 (二元模型)**:
   
```
python main.py <data/input.txt >data/output.txt 
或者
python main.py -n 2 <data/input.txt >data/output.txt 
```
包含同时预处理语料库和运行拼音输入法

如果已经预处理过语料库,只想运行输入法，运行
   
```
python main.py -m eval <data/input.txt >data/output.txt 
或者
python main.py -n 2 -m eval <data/input.txt >data/output.txt 
```
2. **三元模型**:
   
```

python main.py -n 3 <data/input.txt >data/output.txt 
```
包含同时预处理语料库和运行拼音输入法

如果已经预处理过语料库,只想运行输入法，运行
   
```
python main.py -n 3 -m eval <data/input.txt >data/output.txt 
```
3.**评测**
运行`python main.py -m cmp`将评测output.txt相对answer.txt的字准确率和句准确率