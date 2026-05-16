import json
import re
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from pypinyin import pinyin,Style
#将收集到的语料库写到materials.txt
def load_materials():
    #收集到的语料
    datas = []
    materials = None
    folder_path = Path("corpus/sina_news_gbk/")
    for path in folder_path.glob('2*.txt'):
        df = pd.read_json(path,lines=True,encoding='gbk')
        df['combined'] = df['title']+'\n'+df['html']
        datas.extend(df.combined.tolist())
    materials = "\n".join(datas).strip()
    with open("materials.txt",'w',encoding='utf-8',errors = 'ignore') as f:
        f.write(materials)
#从生成的materials.txt中得到oj格式的一、二元词频表
def generate_ngram_count():
    #从一二级汉字表得到合法汉字集
    valid_char = set()
    with open("data/一二级汉字表.txt",'r',encoding="utf-8") as f:
        content = f.read()
        for c in content:
            valid_char.add(c)
    #单字字频
    unigram_counts = Counter()
    #二元词词频
    bigram_counts = Counter()
    #统计单字字频
    with open("materials.txt",'r',encoding="utf-8") as f:
        for line in tqdm(f, desc="处理语料", unit="行" ):
            hanzi_blocks = re.split(r'[^\u4e00-\u9fa5]+',line)
            for block in hanzi_blocks:
                if not block:
                    continue
                #只保留汉字表中的汉字
                clean_block = "".join([c for c in block if c in valid_char])
                #统计1_word
                for hanzi in clean_block:
                    unigram_counts[hanzi]+=1
                #统计2_word
                for iter in range(len(clean_block)-1):
                    pair = "".join(clean_block[iter:iter+2])
                    bigram_counts[pair]+=1
                
    #oj格式一元、二元词表
    uniword = {}
    biword = {}
    #汉字和对应的拼音
    hz_py = defaultdict(list)
    #加载拼音汉字表并获得oj格式的1_word.txt
    with open("data/拼音汉字表.txt") as f:
        for line in f:
            
            parts = line.strip().split()
            if len(parts)>1:
                py ,words= parts[0],parts[1:]
                if py not in uniword:
                    uniword[py] = {"words": [], "counts": []}
                for hanzi in words:
                    hz_py[hanzi].append(py)
                    count = unigram_counts.get(hanzi, 0)
                    if count > 0:  # 只添加在语料中出现过的汉字
                        uniword[py]["words"].append(hanzi)
                        uniword[py]["counts"].append(count)
    #获得oj格式的2_word.txt
    for word_pair, count in bigram_counts.most_common(int(1e6)):
        if word_pair[0] in hz_py and word_pair[1] in hz_py:
            for i in range(len(hz_py[word_pair[0]])):
                for j in range(len(hz_py[word_pair[1]])):
                    pair_pinyin = f"{hz_py[word_pair[0]][i]} {hz_py[word_pair[1]][j]}"
                    if pair_pinyin not in biword:
                        biword[pair_pinyin] = {"words": [], "counts": []}
                    bi_words = f"{word_pair[0]} {word_pair[1]}"
                    biword[pair_pinyin]["words"].append(bi_words)
                    biword[pair_pinyin]["counts"].append(count)
    #将一元、二元字表写入1_word.txt
    with open("1_word.txt" ,'w' , encoding="utf-8") as f:
        json.dump(uniword, f, indent=4, ensure_ascii = False)
   
    with open("2_word.txt", 'w', encoding="utf-8") as f:
        json.dump(biword, f, indent = 4, ensure_ascii=False)
                
#利用pypinyin库标注拼音生成1_word.txt和2_word.txt
def generate_ngram_count_utilize_pypinyin():
    #从一二级汉字表得到合法汉字集
    valid_char = set()
    with open("data/一二级汉字表.txt",'r',encoding="utf-8") as f:
        content = f.read()
        for c in content:
            valid_char.add(c)
    #单字字频
    unigram_counts = Counter()
    #二元词词频
    bigram_counts = Counter()
    #统计单字字频
    with open("materials.txt",'r',encoding="utf-8") as f:
        for line in tqdm(f, desc="处理语料", unit="行" ):
            hanzi_blocks = re.split(r'[^\u4e00-\u9fa5]+',line)
            for block in hanzi_blocks:
                if not block:
                    continue
                #只保留汉字表中的汉字
                clean_block = "".join([c for c in block if c in valid_char])
                #统计1_word
                for hanzi in clean_block:
                    unigram_counts[hanzi]+=1
                #统计2_word
                for iter in range(len(clean_block)-1):
                    pair = "".join(clean_block[iter:iter+2])
                    bigram_counts[pair]+=1
                
    #oj格式一元、二元词表
    uniword = {}
    biword = {}
    #汉字和对应的拼音
    hz_py = defaultdict(list)
    #加载拼音汉字表并获得oj格式的1_word.txt
    with open("data/拼音汉字表.txt") as f:
        for line in f:
            
            parts = line.strip().split()
            if len(parts)>1:
                py ,words= parts[0],parts[1:]
                if py not in uniword:
                    uniword[py] = {"words": [], "counts": []}
                for hanzi in words:
                    hz_py[hanzi].append(py)
                    count = unigram_counts.get(hanzi, 0)
                    if count > 0:  # 只添加在语料中出现过的汉字
                        uniword[py]["words"].append(hanzi)
                        uniword[py]["counts"].append(count)
    #获得oj格式的2_word.txt
    for word_pair, count in bigram_counts.most_common(int(1e6)):
        if word_pair[0] in hz_py and word_pair[1] in hz_py:
            #利用pypinyin标注二元词组拼音
            pinyins = pinyin(word_pair,style=Style.NORMAL) #[['ji'], ['shu']]
            for i in range(len(pinyins[0])):
                for j in range(len(pinyins[1])):
                    pair_pinyin = f"{pinyins[0][i]} {pinyins[1][j]}"
                    if pair_pinyin not in biword:
                        biword[pair_pinyin] = {"words": [], "counts": []}
                    bi_words = f"{word_pair[0]} {word_pair[1]}"
                    biword[pair_pinyin]["words"].append(bi_words)
                    biword[pair_pinyin]["counts"].append(count)
           
    #将一元、二元字表写入1_word.txt
    with open("1_word.txt" ,'w' , encoding="utf-8") as f:
        json.dump(uniword, f, indent=4, ensure_ascii = False)
   
    with open("2_word.txt", 'w', encoding="utf-8") as f:
        json.dump(biword, f, indent = 4, ensure_ascii=False)

#利用pypinyin库标注拼音生成3_word.txt
def generate_3gram_count_utilize_pypinyin():
    #从一二级汉字表得到合法汉字集
    valid_char = set()
    with open("data/一二级汉字表.txt",'r',encoding="utf-8") as f:
        content = f.read()
        for c in content:
            valid_char.add(c)
    trigram_counts = Counter()
    with open("materials.txt",'r',encoding="utf-8") as f:
        for line in tqdm(f, desc="处理语料", unit="行" ):
            hanzi_blocks = re.split(r'[^\u4e00-\u9fa5]+',line)
            for block in hanzi_blocks:
                if not block:
                    continue
                #只保留汉字表中的汉字
                clean_block = "".join([c for c in block if c in valid_char])
                #统计3_word
                for iter in range(len(clean_block)-2):
                    pair = "".join(clean_block[iter:iter+3])
                    trigram_counts[pair]+=1
    #获得oj格式的2_word.txt
    triword = {}
    for word_pair, count in trigram_counts.most_common(int(1e6)):
        
        #利用pypinyin标注二元词组拼音
        pinyins = pinyin(word_pair,style=Style.NORMAL) #[['ji'], ['shu'], ['liu']]
        for i in range(len(pinyins[0])):
            for j in range(len(pinyins[1])):
                for k in range(len(pinyins[2])):
                    pair_pinyin = f"{pinyins[0][i]} {pinyins[1][j]} {pinyins[2][k]}"
                    if pair_pinyin not in triword:
                        triword[pair_pinyin] = {"words": [], "counts": []}
                    tri_words = f"{word_pair[0]} {word_pair[1]} {word_pair[2]}"
                    triword[pair_pinyin]["words"].append(tri_words)
                    triword[pair_pinyin]["counts"].append(count)
    with open("3_word.txt",'w',encoding="utf-8") as f:
        json.dump(triword,f,indent=4,ensure_ascii=False)
if __name__ == "__main__":
    #load_materials()
    #generate_ngram_count_utilize_pypinyin()
    generate_3gram_count_utilize_pypinyin()
    