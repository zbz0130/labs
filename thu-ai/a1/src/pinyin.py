import sys
import json
import math
SMALL = 0.001
#加载数据
def load_data():
    with open('./1_word.txt','r',encoding = 'utf-8') as f:
        unigram = json.load(f)
    with open('./2_word.txt','r',encoding = 'utf-8') as f:
        bigram = json.load(f)
    #unigram_dict["qing"]["清"] = freq
    unigram_dict = {}
    #bigram_dict["ba nen"]["巴 嫩"] = 548
    bigram_dict = {}
    #统计拼音中词的总频数
    #unigram_nums["qing"] = 1000
    unigram_nums = {}
    tot_nums = 0
    for pinyin,datas in unigram.items():
        py_dict = {}
        for word,counts in zip(datas["words"], datas["counts"]):
            py_dict[word] = counts
            unigram_nums[pinyin]=unigram_nums.get(pinyin,0)+counts
            tot_nums+=counts
        unigram_dict[pinyin] = py_dict
    for pinyinpair, datas in bigram.items():
        pair_dict = {}
        for words, counts in zip(datas["words"], datas["counts"]):
            pair_dict[words] = counts
            # bigram_nums[pinyinpair]+=counts
        bigram_dict[pinyinpair] = pair_dict
    return unigram_dict, bigram_dict, unigram_nums, tot_nums

#拼音->汉字
"""
P(w1w2...wk) = P(w1)P(w2|w1)P(w3|w1w2)...P(wk|w1w2w3..wk-1)
\approx P(w1)P(w2|w1)P(w3|w2)....P(wk|wk-1)
"""
def viterbi(pinyins, unigram_dict, bigram_dict, unigram_nums, tot_nums):
    LAMBDA = 0.99 #平滑概率参数
    def get_uni_prob(py,ch):
        """
        P(ch)的概率
        Args:
            py :拼音
            ch :字符
        """
        return unigram_dict.get(py,{}).get(ch,0.01)/tot_nums
    #dp[轮数][字] = 概率
    dp = [{} for _ in range(len(pinyins))]
    #回溯指针
    backptr = [{} for _ in range(len(pinyins))]
    FirstChrs = unigram_dict.get(pinyins[0],{}).keys()
    p0 = 0.0
    best_score = -float('inf')
    for chr0 in FirstChrs:
        # p0 = unigram_dict[pinyins[0]][chr0] / unigram_nums[pinyins[0]]
        p0 = unigram_dict[pinyins[0]][chr0] / tot_nums
        dp[0][chr0] = math.log(p0)
        backptr[0][chr0] = None
    for itr in range(1,len(pinyins)):
        py_prev = pinyins[itr-1]
        py_curr = pinyins[itr]
        py_pair = " ".join([py_prev,py_curr])
        candidates_prev = unigram_dict.get(py_prev,{}).keys()
        candidates_curr = unigram_dict.get(py_curr,{}).keys()
        for chr_curr in candidates_curr:
            dp[itr][chr_curr] = -float('inf')
            for chr_prev in candidates_prev:
                p_uni = get_uni_prob(py_curr, chr_curr)
                word_pair = f"{chr_prev} {chr_curr}"
                prev_count = unigram_dict.get(py_prev, {}).get(chr_prev, 0)
                if prev_count > 0:
                    bi_count = bigram_dict.get(py_pair, {}).get(word_pair, 0)
                    p_bi = LAMBDA * bi_count / prev_count + (1 - LAMBDA) * p_uni
                else:
                    p_bi = p_uni
                if p_bi <=0:
                    p_bi  = SMALL
                dp[itr][chr_curr] = max(dp[itr-1][chr_prev]+math.log(p_bi),dp[itr][chr_curr])
                if dp[itr][chr_curr] == dp[itr-1][chr_prev]+math.log(p_bi):
                    backptr[itr][chr_curr] = chr_prev
    n = len(pinyins) - 1
    best_ch = None
    for ch, score in dp[n].items():
        best_score = max(best_score, score)
        if best_score == score:
            best_ch =  ch
    result = []
    for t in range(n,-1,-1):
        result.append(best_ch)
        best_ch = backptr[t][best_ch]
    result.reverse()
    return "".join(result)

def pinyin():
    try:
        unigram_dict, bigram_dict, unigram_nums,tot_nums = load_data()
    except Exception as e:
        print(f"数据加载有问题：{e}")
        sys.exit(1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        pinyins = line.split()
        sentence = viterbi(pinyins,unigram_dict, bigram_dict,unigram_nums,tot_nums)
        print(sentence)
if __name__ == "__main__":
    pinyin()