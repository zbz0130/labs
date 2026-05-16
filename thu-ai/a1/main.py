import argparse
from src.pinyin import pinyin
from src.data_process import load_materials,generate_ngram_count\
    ,generate_ngram_count_utilize_pypinyin ,generate_3gram_count_utilize_pypinyin
from src.pinyin_3w import pinyin_3gram
from src.compare import compare
def main(ngram,mode):
    if mode == "cmp":
        compare()
        return
    if mode=="train" :
        load_materials()
        generate_ngram_count_utilize_pypinyin()
    if ngram==2:
        pinyin()
        return
    elif ngram==3:
        if mode=="train" :
            generate_3gram_count_utilize_pypinyin()
        pinyin_3gram()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pinyin Input Method(N-gram)")
    parser.add_argument("-n", "--ngram", type = int, choices = [2,3], default = 2, help = "选择模型阶数:2代表二元模型,3代表3元模型")
    parser.add_argument("-m","--mode",type = str,choices = ["train", "eval", "cmp"], default = "train", help = "选择运行方式:train代表训练+运行,eval代表只运行" )
    args = parser.parse_args()
    main(args.ngram,args.mode)
