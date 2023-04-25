import random
import glob
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, BertForMaskedLM, BertModel
import sys
import pandas as pd
import re

import bert_score
# hide the loading messages
import logging

import MeCab
import subprocess
from IPython.display import HTML
from func1 import *
import transformers
import matplotlib.pyplot as plt
from matplotlib import rcParams

import json
import emoji
import mojimoji
import neologdn
import nltk
from nltk import word_tokenize
from nltk import bleu_score
import scipy.spatial
import scipy


class CleansingTweets:

    # replace and\s to space
    def cleansing_space(self, text):
        return re.sub("\u3000|\s", " ", text)

    #repeat string abbrebiation
    def cleansing_repeat(self,text):
        text = re.sub("!{2,}","!",text)
        text = re.sub("\?{2,}","?",text)
        text = re.sub("(!\?){2,}","!?",text)
        text = re.sub("w{2,}","w",text)
        text = re.sub("…{2,}","…",text)
        text = text.replace("〝","\"")
        
        return text
    
    # remove hashtags
    def cleansing_hash(self, text):
        return re.sub("#[^\s]+", "", text)

    # remove URLs
    def cleansing_url(self, text):
        return re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" , text)

    # remove pictographs
    def cleansing_emoji(self, text):
        return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)

    # remove mentions
    def cleansing_username(self, text):
        return re.sub(r"@([A-Za-z0-9_]+) ", "", text)

    #  remove image strings
    def cleansing_picture(self, text):
        return re.sub(r"pic.twitter.com/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]*", "" , text)

    # unify characters
    def cleansing_unity(self, text):
        text = text.lower()
        text = mojimoji.zen_to_han(text, kana=True)
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        return text

    # replace number to zero
    def cleansing_num(self, text):
        text = re.sub(r'\d+', "0", text)
        return text

    # remove rt
    def cleansing_rt(self, text):
        return re.sub(r"RT @[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]*?: ", "" , text)

    def cleansing_text(self, text):
        text = self.cleansing_rt(text)
        text = self.cleansing_hash(text)
        text = self.cleansing_space(text)
        text = self.cleansing_url(text)
        text = self.cleansing_emoji(text)
        text = self.cleansing_username(text)
        text = self.cleansing_picture(text)
        text = self.cleansing_unity(text)
        #text = self.cleansing_num(text)
        text = neologdn.normalize(text)
        text = self.cleansing_repeat(text)
        return text

    def cleansing_df(self, df, subset_cols=["text"]):
        if "text" in subset_cols:
            # remove duplicates (because they might be RT.)
            df = df.drop_duplicates(subset="text", keep=False)

        df_copy = df.copy()

        for col in subset_cols:
            # cleansing
            df_copy[col] = df[col].apply(lambda x: self.cleansing_text(x))

        if "text" in subset_cols:
            # remove duplicates
            df_copy = df_copy.drop_duplicates(subset="text", keep=False).reset_index(drop=True)

        return df_copy
    
#csvをusecolsの場所でpandasの形で抽出する    
def makeclean_csv(name, usecol_num=1, cols_name = "text"):
    df = pd.read_csv(name + ".csv",encoding="utf-8",engine="python")
    
    tweet_cleaner = CleansingTweets()
    cols = [cols_name]
    df_clean = tweet_cleaner.cleansing_df(df,subset_cols=cols)
    #もし重複があれば削除する
    is_complete_duplicate = (df.duplicated(keep=False))
    is_complete_duplicate
    df_complete_duplicate = df[is_complete_duplicate]
    df_complete_duplicate
    print(df_clean)
    return df_clean

#この研究に合わせたタグ取得方法
def tag_get(name):
    #load data
    df = pd.read_csv(name + ".csv",encoding="utf-8",engine="python")
    text_lis = []
    f_text_lis = []
    s_text_lis = []
    for text,tag in zip(df["text"],df["tag"]):
        if tag == 0:
            text_lis.append(text)
        elif tag == 1:
            f_text_lis.append(text)
        else:
            s_text_lis.append(text)
    
    return text_lis,f_text_lis,s_text_lis

#データの符号化
def Encoding(text_list, labels, max_length=128):
    dataset_for_loader = []
    for idx,text in enumerate(text_list):
        encoding = tokenizer(text, max_length=max_length,padding="max_length",truncation=True) #textを形態素解析、"pt"でtensor出力,辞書型でreturn
        encoding["labels"] = labels[idx] #add label
        encoding = {k: torch.tensor(v) for k,v in encoding.items()}  
        dataset_for_loader.append(encoding)
    return dataset_for_loader

def random_list(lis,seed=42):
    random.seed()
    random.shuffle(lis)
    return lis

#データセットの分割
def dataset_separate(dataset):
    n = len(dataset)
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    dataset_train = dataset[:n_train]
    dataset_val = dataset[n_train:n_train + n_val]
    dataset_test = dataset[n_train+n_val:]
    return dataset_train, dataset_val, dataset_test

#mecabによる形態素解析(ipadic-neologd),固有名詞の抽出
def extract_NER(text):
    cmd = 'echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
    path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True).communicate()[0]).decode('utf-8')
    m = MeCab.Tagger("-d {0}".format(path))  #mecab-ipadic-neologd
    #m = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    node = m.parse(text)
    node = node.split("\n")
    text_NER = []
    for i in range(len(node)):
        word = node[i].split()
        if word[0] == "EOS":
            break
        word2 = word[1].split(",")
        try:
            if word2[1] == "固有名詞":
                text_NER.append(word[0])
        except:
            print("Error NER")
    return text_NER

def extract_notreplace_words(text):
    cmd = 'echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
    path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True).communicate()[0]).decode('utf-8')
    m = MeCab.Tagger("-d {0}".format(path))  #mecab-ipadic-neologd
    #m = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    node = m.parse(text)
    node = node.split("\n")
    text_NER = []
    for i in range(len(node)):
        word = node[i].split()
        if word[0] == "EOS":
            break
        word2 = word[1].split(",")
        try:
            if word2[0] == "助詞" or word2[0] == "助動詞" or word2[1] == "固有名詞":
                text_NER.append(word[0])
        except:
            print("Error NER")

    return text_NER

#wordpiecesにて分割されている単語を復元する,対応するattentionの値も調整
def word_connection(word_lis,attn_lis):
    return_word = []
    return_attn = []
    a = 0
    ab = 0
    times = 1
    p_times = 1
    back_times = 1
    p_back_times = 1
    for idx, (word, att) in enumerate(zip(word_lis,attn_lis)):
        if word[0] == "#":  #初めが#のとき
            s = return_word.pop()
            a += att
            string = s + word.replace("##","")
            times += 1
            return_word.append(string)
        else:  
            if times == p_times and idx != 0 and back_times == 1:
                a += return_attn.pop()
                attn = a / times
                return_attn.append(attn)
                a = 0
                times = 1
                
            if word_lis[idx-1][-1] == "#":  #一つ前の要素がうしろで#があるとき
                s = return_word.pop()
                ab += att
                string = s.replace("##","") + word
                return_word.append(string)
                back_times += 1
                
            else:  #通常表示のとき
                return_word.append(word)
                attn = (ab+att) / back_times
                return_attn.append(attn)
                back_times = 1
                ab = 0
            
        p_times = times
        p_back_times = back_times
                
    return return_word, return_attn

#純粋な最大最小正規化法
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

#最小値を渡せば動作する最大最小正規化法
def min_max2(x,min, axis=None):
    max = x.max(axis=axis, keepdims=True)
    result = (x-min) / (max-min)
    return result

def highlight(word, attn):
    #Attention の値が大きいと文字の背景が濃い赤になる html を出力させる関数
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)

#main, open sentence of attention, pick up attention words
def mk_around(text, attn_weights,tokenizer, max_length=128, printout=False):
    #HTMLデータの作成,12層のみの表示
    """
    index: 表示したいデータ番号
    sentences: 可視化したい文章
    pred_ls: 予測したラベルデータ
    attn_weights: attentionの重みデータ
    """
    sentence = tokenizer.convert_ids_to_tokens(text)
    
    #12層目のAttention層飲みの表示システム
    sentence, attention = word_connection(sentence,attn_weights[0,11,0,:].tolist())
    #最大最小値法による正規化
    #############ここから#################
    atten_remove_zero = [x for x in attention if x != 0]
    atten_remove_zero = np.array(atten_remove_zero)
    min = atten_remove_zero.min(axis=None, keepdims=True)
    #print(min)
    attention = np.array(attention)
    attention = min_max2(attention,min=min)
    #############ここまで#################
    attention = torch.tensor(attention).cuda()
    #最大値による正規化#####################
    #attention /= attention.max()
    #####################
    at = attention.tolist()
    t = sentence[:sentence.index("[SEP]")]
    #自分で考えて実験する値
    #num = len(t) / -340 + 0.92 ##計算式 (len) / -340 + 0.92
    num = 0.82 #magic number
    if printout == True:
        print(num)                                                        ############################attention適用値確認
    word_index = index_multi(at,x=num)  #この値を調整することで収集できる語句が大きく変化するため要調整　0.82
    #print(word_index)
    flaming_word_dic = {}
    for idx in word_index:
        if printout == True:
            print(idx,at[idx])                                               ###########################attentionの値確認用
        if idx == 0:
            continue
        else:
            ######pattern even use
            if idx > 1:
                flaming_word_dic[idx-1] = sentence[idx-1]
            if idx < len(sentence) - 1:
                flaming_word_dic[idx+1] = sentence[idx+1] 
             #####pattern even end
            #pass # pattern 1,3,5, else use around get
        flaming_word_dic[idx] = sentence[idx]
    
    return flaming_word_dic

"""
#attentionによる注目語句の抽出
def attention_words(text,tokenizer,bert_sc, printout = False):
    encoding = tokenizer(text,max_length=128,padding="max_length",return_tensors="pt")#形態素解析、tensor出力
    encoding = {k : v.cuda() for k,v in encoding.items()}
    with torch.no_grad():
        output = bert_sc.forward(**encoding)
    attention = output.attentions
    attention = next(iter(attention))
    sen = encoding["input_ids"]
    danger_dic = mk_html(sen[0],attention,tokenizer)
    if printout == True:
        print(danger_dic)                                                ######################ここがdic確認(アテンション)
    
    return danger_dic
"""

def at_words_around(text,tokenizer,bert_sc,printout = False):
    """
    Attentionによって得た語句のその周辺の単語に対してもMASK処理を行って予測を行うものである。
    """
    encoding = tokenizer(text,max_length=128,padding="max_length",return_tensors="pt")#形態素解析、tensor出力
    encoding = {k : v.cuda() for k,v in encoding.items()}
    with torch.no_grad():
        output = bert_sc.forward(**encoding)
    attention = output.attentions
    attention = next(iter(attention))
    sen = encoding["input_ids"]
    danger_dic = mk_around(sen[0],attention,tokenizer)
    if printout == True:
        print(danger_dic)                                                ######################ここがdic確認(アテンション)
    
    return danger_dic


def calc_bert_score(cands,refs):
    Precision,Recall,F1 = bert_score.score(cands, refs, lang="ja", verbose=True)
    #P, R, F1 = bert_score.score(cands, refs, lang='ja', rescale_with_baseline=True)
    return F1.numpy().tolist()

def textdata_load(name):
    #load data
    df = pd.read_csv(name + ".csv",encoding="utf-8",engine="python")
    text_lis = []
    for text in df["text"]:
        text_lis.append(text)
    
    return text_lis

def index_multi(lis, x):
    return [i for i, _x in enumerate(lis) if _x >= x]

def predict_mask_topk(text, tokenizer, bert_mlm, num_topk, errormsg_off = False):
    """
    文章中の最初のMASKをスコア上位のトークンに置き換える
    num_topkで数を指定
    出力は穴埋めされた文章のリストと、置き換えられたトークンのスコアのリスト
    """
    #文章を符号化し、BERTで分類スコアを得る
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.cuda()
    #print(input_ids)
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)
    scores = output.logits
    
    #スコアが上位のトークンとスコアを求める
    try:
        mask_position = input_ids[0].tolist().index(4) #MASKがある場所を取得、なければエラーが出る
        topk = scores[0,mask_position].topk(num_topk)
        ids_topk = topk.indices
        tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk)
        scores_topk = topk.values.cpu().numpy()

        #文章中のMASKを上で求めたトークンで置き換える
        text_topk = []
        for token in tokens_topk:
            token = token.replace("##", "") #もしかしたら”　”のように空白が良い可能性もある
            text_topk.append(text.replace("[MASK]", token ,1)) #< + "[MASK]" + >のようにすることで変換場所が可視化できる
    except ValueError : 
        text_topk = []
        if errormsg_off == True:
            print("ValueError: not MASK")
        text_topk.append(text)
        scores_topk = 0
        
    return text_topk, scores_topk

def beam_search(text, tokenizer, bert_mlm, num_topk, danger_lib, replace_NER = True):

    #ビームサーチで文章の穴埋めを行う
    not_replace = ["。", "、", "…", "...", "(", ")", "「", "」", "[", "]", "{", "}","【","】","…","!","?"]
    if replace_NER == True:
        not_replace2 = extract_NER(text)
        not_replace3 = extract_notreplace_words(text)
        not_replace.extend(not_replace2)
        not_replace.extend(not_replace3)
    text_topk = [text] #ここまでで危険表現をまとめた辞書の作成
    scores_topk = np.array([0])
    #[MASK]予測
    for word in danger_lib.values():
        if word in not_replace:
            continue
        text_candidates = []
        score_candidates = []
        sentences = []
        for sen in text_topk:  #MASK処理をする
            sen = sen.replace(word, "[MASK]", 1) #予め抽出しておいた表現をMASKに変換する
            sentences.append(sen) #beamsearchのように回していくためのlist
        for text_mask, score in zip(sentences, scores_topk):  #beamsearchのようなもの
            text_topk_inner, scores_topk_inner = predict_mask_topk(text_mask,tokenizer,bert_mlm,num_topk)
            text_candidates.extend(text_topk_inner)
            score_candidates.append(score + scores_topk_inner)

        #合計スコアの高いものを選択する
        score_candidates = np.hstack(score_candidates)
        idx_list = score_candidates.argsort()[::-1][:num_topk]
        text_topk = [text_candidates[idx] for idx in idx_list]
        scores_topk = score_candidates[idx_list]
        #print(text_candidates)

    return text_topk

def predict_arrange(text,danger_list, printout = False):
    """
    危険辞書による伏字を適用した際、ひとつづつMASK処理を行うための関数
    return形式は {textの危険単語の文字列の位置 : 危険単語}
    """
    danger_lib = {}
    for word in danger_list:
        if word in text:
            place = [m.start() for m in re.finditer(word, text)]
            for num in place:
                danger_lib[num] = word
    danger_lib = sorted(danger_lib.items())
    danger_lib = dict(danger_lib)
    if printout == True:
        print(danger_lib)                                               ################ここがdic確認(危険表現)

    return danger_lib

#times = len(danger_lib)
def predict_one_after_another(text, tokenizer, bert_mlm, bert_sc, num_topk, danger_list,times, attention_process = False):
    """
    研究でおそらく使用していくことになるであろう関数
    danger_lib :　危険表現が{listのindex : word}で保管されている
    danger_list : [MASK]したい語句をまとめたリスト
    """
    #choice = times % 3 #roop回数に応じて変化を出したいため、スコア上位1,2,3,1,2...のようにループさせる(pattern 5-6)
    choice = 0 #if you do pattern 1-4, use this
    if attention_process is True:
        danger_words = at_words_around(text,tokenizer,bert_sc) #危険単語の周辺単語もマスクする #when pattern even(2,4,6) use.
        text_topk_at = beam_search(text, tokenizer, bert_mlm, 3, danger_words) #上位三件までの予測
        if len(text_topk_at) != 1:
            text = text_topk_at[choice]
        else:
            text = text_topk_at[0]
    #"""
    return text
        
    #pattern 3-6
    """    
    danger_lib = predict_arrange(text,danger_list) # pattern 3-6 use
    text_topk = beam_search(text,tokenizer,bert_mlm,3, danger_lib) #pattern 3-6 use
    if len(text_topk) != 1:
        return_text = text_topk[choice]
    else:
        return_text = text_topk[0]
    """
   
    #return return_text

def labeling_flaming(name, bert_sc, tokenizer):
    #無作為に所得したデータに対するチューニング済みモデルによる判定
    text_list = textdata_load(name)
    text_list = text_list[:2000]
    
    encoding = tokenizer(text_list,padding="longest",return_tensors="pt")#形態素解析、tensor出力
    encoding = {k : v.cuda() for k,v in encoding.items()}
    
    #推論
    with torch.no_grad():
        output = bert_sc.forward(**encoding)
    scores = output.logits
    labels_predicted = scores.argmax(-1)
    #炎上文切り抜きのための実装
    p_labels = labels_predicted.tolist()
    flaming_idx = index_multi(p_labels,1)
    #print(flaming_idx)
    flaming_text = []
    for idx in flaming_idx:
        flaming_text.append(text_list[idx])
    
    df3 = pd.DataFrame({"flaming_text" : flaming_text})
    df3.to_csv("judge_flaming_ex.csv")
    
    return df3

#pandasのdfに対して行う
def labeling_flaming_df(df, bert_sc, tokenizer):
    #無作為に所得したデータに対するチューニング済みモデルによる判定
    
    text_list = []
    for text in df["safety_text"]:
        text_list.append(text)
    
    encoding = tokenizer(text_list,padding="longest",return_tensors="pt")#形態素解析、tensor出力
    encoding = {k : v.cuda() for k,v in encoding.items()}
    
    #推論
    with torch.no_grad():
        output = bert_sc.forward(**encoding)
    scores = output.logits
    labels_predicted = scores.argmax(-1)
    #炎上文切り抜きのための実装
    p_labels = labels_predicted.tolist()
    flaming_idx = index_multi(p_labels,1)
    #print(flaming_idx)
    flaming_text = []
    for idx in flaming_idx:
        flaming_text.append(text_list[idx])
    
    df3 = pd.DataFrame({"flaming_text" : flaming_text})
    df3.to_csv("judge_flaming_ex.csv")
    
    return df3, flaming_idx

def flaming_to_safety(tokenizer, bert_mlm, bert_sc, num_topk, times, attention_process):
    """
    tokenizer : 形態素解析
    bert_mlm : Bertのmlm
    bert_sc : Bertのsc
    num_topk : beamsearchの最大保持数
    attention_process : <bool> attentionの過程をTrueで表示する
    """
    with open("all_flaming_re.txt","r",encoding="UTF-8") as f:
        danger_list = [s.strip() for s in f.readlines()]
    
    df = pd.read_csv("judge_flaming_ex.csv",encoding="utf-8",engine="python",usecols=[1])
    predict_text = []
    flamimg_text = []
    for i,text in tqdm(enumerate(df["flaming_text"])):
        flamimg_text.append(text)
        ans_text = predict_one_after_another(text,tokenizer,bert_mlm,bert_sc,num_topk,danger_list,times, attention_process=attention_process)
        #print(ans_text)
        predict_text.append(ans_text)
    df = pd.DataFrame({"flaming_text" : flamimg_text, "safety_text" : predict_text})
    df.to_csv("bert_textlist_ex.csv")
    return df

def BERTscore_show():
    """
    BERTScore show
    bert_texxtlistにあるsafety_textとflaming_textを抽出
    model_result.csvにF値(類似度)を載せた結果を加える
    """
    rcParams["xtick.major.size"] = 0
    rcParams["xtick.minor.size"] = 0
    rcParams["ytick.major.size"] = 0
    rcParams["ytick.minor.size"] = 0

    rcParams["axes.labelsize"] = "large"
    rcParams["axes.axisbelow"] = True
    rcParams["axes.grid"] = True

    df = pd.read_csv("bert_textlist_ex.csv",encoding="utf-8",engine="python",index_col=0)
    before_text = []
    after_text = []
    for text in df["flaming_text"]:
        before_text.append(text)
    for text in df["safety_text"]:
        after_text.append(text)
    
    F1 = calc_bert_score(before_text,after_text)

    plt.hist(F1)  #bin=num で表示数の数を制限できる
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()
    df["F1"] = F1
    df.to_csv("model_result_ex.csv")
    print(sum(F1) / len(F1)) #BERTScore
    return df

#比べる文章ふたつをdfの形で並べたものによる比較
def BERTscore_show_df(df,num=2):
    """
    BERTScore show
    bert_texxtlistにあるsafety_textとflaming_textを抽出
    model_result.csvにF値(類似度)を載せた結果を加える
    """
    rcParams["xtick.major.size"] = 0
    rcParams["xtick.minor.size"] = 0
    rcParams["ytick.major.size"] = 0
    rcParams["ytick.minor.size"] = 0

    rcParams["axes.labelsize"] = "large"
    rcParams["axes.axisbelow"] = True
    rcParams["axes.grid"] = True

    before_text = []
    after_text = []
    for text in df["flaming_text"]:
        before_text.append(text)
    for text in df["safety_text"]:
        after_text.append(text)
    
    F1 = calc_bert_score(before_text,after_text)

    plt.hist(F1)  #bin=num で表示数の数を制限できる
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()
    df["F1"] = F1
    df.to_csv("model_result" + str(num) + ".csv")
    print("avarage score: ",sum(F1) / len(F1)) #BERTScore
    return df

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
    
def senBERTscore(df,num):
    rcParams["xtick.major.size"] = 0
    rcParams["xtick.minor.size"] = 0
    rcParams["ytick.major.size"] = 0
    rcParams["ytick.minor.size"] = 0

    rcParams["axes.labelsize"] = "large"
    rcParams["axes.axisbelow"] = True
    rcParams["axes.grid"] = True

    model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")
    before_text = []
    after_text = []
    for text in df.iloc[:,0]:
        before_text.append(text)
    for text in df.iloc[:,1]:
        after_text.append(text)
        
    q = model.encode(before_text)
    target = model.encode(after_text).numpy()

    closest_n = 1
    score_lis = []
    for query, query_embedding in zip(after_text, target):
        distances = scipy.spatial.distance.cdist([query_embedding], q, metric="cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    for idx, distance in results[0:closest_n]:
        score_lis.append((1-(distance / 2)))
        
    plt.hist(score_lis)  #bin=num で表示数の数を制限できる
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()
    df["senBERTscore"] = score_lis
    df.to_csv("model_result" + str(num) + ".csv")
    print("avarage score: ",sum(score_lis) / len(before_text)) #senBERT
    return df

#mecabによる形態素解析(ipadic-neologd),taggerを変えれば何でもok
def tokenize(text):
    cmd = 'echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
    path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True).communicate()[0]).decode('utf-8')
    m = MeCab.Tagger("-d {0}".format(path))  #mecab-ipadic-neologd
    #m = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    node = m.parse(text)
    node = node.split("\n")
    text_tokenize = []
    for i in range(len(node)):
        word = node[i].split("\t")
        if word[0] == "EOS":
            break
        text_tokenize.append(word[0])
    
    return text_tokenize

def Bleuscore(df,num):
    before_text = []
    after_text = []
    for text in df.iloc[:,0]:
        before_text.append(text)
    for text in df.iloc[:,1]:
        after_text.append(text)
        
    score_lis = []
    for i in range(len(before_text)):
        hyp = tokenize(before_text[i])
        ref = tokenize(after_text[i])
        #print(hyp)
        ##print("--"*50)
        #print(ref)
        score = bleu_score.sentence_bleu([ref],hyp)
        score_lis.append(score)
    
    rcParams["xtick.major.size"] = 0
    rcParams["xtick.minor.size"] = 0
    rcParams["ytick.major.size"] = 0
    rcParams["ytick.minor.size"] = 0

    rcParams["axes.labelsize"] = "large"
    rcParams["axes.axisbelow"] = True

    rcParams["axes.grid"] = True

    plt.hist(score_lis)  #bin=num で表示数の数を制限できる
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()
    df["Bleuscore"] = score_lis
    df.to_csv("model_result" + str(num) + ".csv")
    print("avarage score: ",sum(score_lis) / len(before_text)) #Bleu
    return df

def Score_all(df,num):
    rcParams["xtick.major.size"] = 0
    rcParams["xtick.minor.size"] = 0
    rcParams["ytick.major.size"] = 0
    rcParams["ytick.minor.size"] = 0

    rcParams["axes.labelsize"] = "large"
    rcParams["axes.axisbelow"] = True

    rcParams["axes.grid"] = True
    
    
    model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")
    before_text = []
    after_text = []
    for text in df["flaming_text"]:
        before_text.append(text)
    for text in df["safety_text"]:
        after_text.append(text)
        
    q = model.encode(before_text)
    target = model.encode(after_text).numpy()

    closest_n = 1
    score_lis = []
    for query, query_embedding in zip(after_text, target):
        distances = scipy.spatial.distance.cdist([query_embedding], q, metric="cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in results[0:closest_n]:
            score_lis.append((1-(distance / 2)))
        
    #print(score_lis)
    print("senBERTscore")
    plt.hist(score_lis)  #bin=num で表示数の数を制限できる
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()
    df["senBERTscore"] = score_lis
    print("avarage score: ",sum(score_lis) / len(before_text))
    
    score_lis = []
    for i in range(len(before_text)):
        hyp = tokenize(before_text[i])
        ref = tokenize(after_text[i])
        #print(hyp)
        ##print("--"*50)
        #print(ref)
        score = bleu_score.sentence_bleu([ref],hyp)
        score_lis.append(score)
    
    print("Bleuscore")
    plt.hist(score_lis)  #bin=num で表示数の数を制限できる
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()
    df["Bleuscore"] = score_lis
    print("avarage score: ",sum(score_lis) / len(before_text))
    
    F1 = calc_bert_score(before_text,after_text)
    
    print("F1score")
    plt.hist(F1)  #bin=num で表示数の数を制限できる
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()
    df["F1"] = F1
    df.to_csv("model_result_ex" + str(num) + ".csv")
    print("avarage score: ",sum(F1) / len(F1)) #BERTScore
    return df