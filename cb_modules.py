## for coldstart plylst

from khaiii import KhaiiiApi

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel


def title_to_token(train_data):
    api = KhaiiiApi()
    title_to_token = []
    vocab = []
    
    for idx in range(len(train_data)):
        token_lst = []
        try:
            title = train_data[idx]['plylst_title']
            for tok in api.analyze(title):
                for morph in tok.morphs:
                    if morph.tag in ['NNG', 'NNP', 'NP', 'MAG', 'MAJ', 'JKS', 'VV', 'VA', 'IC', 'SN', 'SL']:
                        vocab.append(morph.lex)
                        token_lst.append(morph.lex)             
        except:
            vocab.append('.')
            token_lst.append('.')   
            
        title_to_token.append(token_lst)
        
    return title_to_token, vocab


def get_fin_vocab(vocab):
    fin_vocab = {}
    
    c_vocab = Counter(vocab)
    cnt = len(fin_vocab)
    for voca in c_vocab:
        if (voca != '.') and (c_vocab[voca] > 1):
            fin_vocab[voca] = cnt
            cnt += 1
            
    return fin_vocab


def tok_to_idx(tok_plylst, fin_vocab):
    res_idx = []
    for tok in tok_plylst:
        if tok in fin_vocab:
            res_idx.append(fin_vocab[tok])
        else:
            pass
        
    return res_idx


def preproc_for_csr(title_to_idx, from_idx):
    user_lst = []
    vocab_lst = []

    for idx, title in enumerate(title_to_idx[from_idx:]):
        uid = from_idx + idx
        uid_lst = []
        for tok in title:
            vocab_lst.append(tok)
            user_lst.append(uid)             

    return user_lst, vocab_lst


def build_tfidf_mat(plylst_tt_mat):
    tfidf_trans = TfidfTransformer()
    tfidf_mat = tfidf_trans.fit_transform(plylst_tt_mat)
    
    return tfidf_mat


def get_sim_plylst(tfidf_mat, given, topn):
    res_sim = dict()
    most_sim_idx = list()
    
    cos_sim = linear_kernel(tfidf_mat[given:given+1], tfidf_mat).flatten()

    for idx, sim in enumerate(cos_sim):
        if (sim < 1) and (sim > 0):
            res_sim[sim] = idx
            
    most_sim = sorted(res_sim, reverse=True)[:topn]
    
    for sim in most_sim:
        most_sim_idx.append(res_sim[sim])
            
    return most_sim_idx   


def gather_cand(train_data, questions, most_sim_lst):
    cands = list()
    for idx in most_sim_lst:
        if idx > len(train_data)-1:
            idx = idx - len(train_data)
            cands.extend(questions[idx]['songs'])
        else:
            cands.extend(train_data[idx]['songs'])
    
    cand_c = Counter(cands)
    cand_lst = [k for (k, v) in cand_c.most_common(200)]
        
    return cand_lst

