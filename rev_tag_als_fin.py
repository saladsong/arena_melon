# -*- coding: utf-8 -*-
from collections import Counter

import fire
from tqdm import tqdm

from arena_util import load_json
from arena_util import write_json
from arena_util import remove_seen
from arena_util import most_popular

import numpy as np
import pandas as pd
import scipy.sparse as sp
import implicit
import time

from cb_modules import *


class Als_with_CB:
    def _song_mp_per_genre(self, song_meta, global_mp):
        res = {}

        for sid, song in song_meta.items():
            for genre in song['song_gn_gnr_basket']:
                res.setdefault(genre, []).append(sid)

        for genre, sids in res.items():
            res[genre] = Counter({k: global_mp.get(int(k), 0) for k in sids})
            res[genre] = [k for k, v in res[genre].most_common(200)]

        return res
  

    ### for tag prediction (aggregate tags per song for plylst)
    
    def _tag_per_song(self, train_meta):
        res = {}

        for lid, plylst in train_meta.items():
            for sid in plylst['songs']:
                for tag in plylst['tags']:
                    res.setdefault(sid, []).append(tag)
                    #res.setdefault(genre, []).append(sid)

        for sid, tags in res.items():
            c = Counter(tags)
            res[sid] = [k for k, v in c.most_common(10)]  

        return res
    
    def _tag_per_plylst(self, plylst, tag_per_song):
        res_tg = []
        res = []
        
        for song in plylst['songs']:
            if song in tag_per_song.keys():
                tags = tag_per_song[song]
                res_tg.extend(tags)
            else:
                pass
    
        c = Counter(res_tg)
        for tag in res_tg:
            res = [k for k, v in c.most_common(100)]  
        
        return res

    
    ### for filtering most popluar 200k (approx 0.3) items 
    
    def _build_vocadict(self, song_set):
        voca_dict = dict()
        voca_dict_t = dict()

        for idx, item in enumerate(song_set):
            voca_dict[item] = idx
            voca_dict_t[idx] = item
    
        return voca_dict, voca_dict_t
    
    def _filter_mp(self, item_seq, song_mp):
        filtered = []
        for song in item_seq:
            if (song in song_mp):
                filtered.append(song)
            else:
                pass
                #filtered.append(1)
            
        return filtered
     
    def _const_filtered_lst(self, train_data, voca_dict, num_users, to_idx=10000, val=False):
        f_song_lst = list()
        f_usr_lst = list()

        for idx in range(to_idx):
            item_seq = train_data[idx]['songs']
            filtered_lst = self._filter_mp(item_seq, voca_dict)
            seq_len = len(filtered_lst)
            f_song_lst.extend(filtered_lst)
        
            if val == True:
                idx = idx + num_users
            f_usr_lst.extend([idx]*seq_len)

        return f_song_lst, f_usr_lst
    
    def _build_item_map(self, f_song_lst):
        item_map = dict()
        num_item = 1   # 0 for unk 

        for i in f_song_lst:
            if i not in item_map:
                if i == 1:
                    item_map[i] = 0
                else:
                    item_map[i] = num_item
                    num_item += 1
    
        item_map_t = dict((v,k) for k,v in item_map.items())
                
        return item_map, item_map_t
    
    
    def _cal_alsmodel(self, idx, num_users, usr_item, model, item_map_t):
        idx_num = num_users + idx
        prev_items = usr_item[idx_num].toarray()
        len_prev = sum(prev_items[0])
        if len_prev > 0:
            rec_lst = model.recommend(idx_num, usr_item.tocsr(), N=150)
            rec_items_o = [item_map_t[k] for k, v in rec_lst]
        else:
            rec_items_o = list()
    
        return rec_items_o
   

    ###

    def _generate_answers(self, song_meta_json, train, questions):
        song_meta = {int(song["id"]): song for song in song_meta_json}
        train_meta = {int(plylst["id"]): plylst for plylst in train}
        
        song_mp_counter, song_mp = most_popular(train, "songs", 200)
        tag_mp_counter, tag_mp = most_popular(train, "tags", 100)
        song_mp_per_genre = self._song_mp_per_genre(song_meta, song_mp_counter)
        
        tag_per_song = self._tag_per_song(train_meta)
        
        
        ## modified for song prediction
        ## pre-processing train set data
        _, song_pop = most_popular(train, "songs", 200000)
        #song_pop = set(song_pop)
        voca_dict, voca_dict_t = self._build_vocadict(song_pop)
        
        # filtering song list 
        num_users = len(train)
        f_song_lst, f_usr_lst = self._const_filtered_lst(train, voca_dict, num_users, to_idx=num_users, val=False)
        num_items = len(set(f_song_lst))
        data_len = len(f_song_lst)
        
        # re-setting index of filtered songs
        item_ids = np.array([voca_dict[i] for i in f_song_lst])
        data = np.ones(data_len)
        rows, cols, data = zip(*set(zip(f_usr_lst, item_ids, data)))
        print('train preproc done', num_items)
        
        ## pre-processing valid/test set data
        v_num_users = len(questions)
        f_song_lst_v, f_usr_lst_v = self._const_filtered_lst(questions, voca_dict, num_users, to_idx=v_num_users, val=True)
        data_len_v = len(f_song_lst_v)
        
        v_item_ids = np.array([voca_dict[i] for i in f_song_lst_v])
        v_data = np.ones(data_len_v)
        v_rows, v_cols, v_data = zip(*set(zip(f_usr_lst_v, v_item_ids, v_data)))
        print('valid preproc done', num_items)
        
        n_rows = rows + v_rows
        n_cols = cols + v_cols
        n_data = data + v_data
        t_num_users = num_users + v_num_users

        usr_item_mat = sp.csr_matrix( (n_data, (n_rows, n_cols)), shape=(t_num_users, num_items) )
        item_usr_mat = usr_item_mat.T
        als_model = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.05, iterations=50)
        #als_model = implicit.bpr.BayesianPersonalizedRanking(factors=50)  ### actually bpr 
        #als_model = implicit.lmf.LogisticMatrixFactorization(factors=50) ### actually Logistic MF
        als_model.fit(item_usr_mat)
        print("als model fitting done")
        
        
        ### for cold-start users (plylst containing no song)
        title_to_tok, vocab = title_to_token(train)
        v_title_to_tok, v_vocab = title_to_token(questions)
        title_to_tok.extend(v_title_to_tok)
        vocab.extend(v_vocab)
        print("title to token converted", len(title_to_tok), len(vocab) )
        
        fin_vocab = get_fin_vocab(vocab)
        print("final vocab size", len(fin_vocab))
        
        title_to_idx = []
        for plylst in title_to_tok:
            res_idx = tok_to_idx(plylst, fin_vocab)
            title_to_idx.append(res_idx)
            
        user_lst, vocab_lst = preproc_for_csr(title_to_idx, 0)
        cb_rows = np.array(user_lst)
        cb_cols = np.array(vocab_lst)
        cb_data = np.ones(len(user_lst))
        plylst_tt_mat = sp.csr_matrix((cb_data, (cb_rows, cb_cols)), shape=(len(title_to_tok), len(fin_vocab)))
        print("csr matrix for tf-idf matrix made")
        
        tfidf_mat = build_tfidf_mat(plylst_tt_mat)
    
     
        ####

        answers = []
        for idx, q in tqdm(enumerate(questions)):
            genre_counter = Counter()

            for sid in q["songs"]:
                for genre in song_meta[sid]["song_gn_gnr_basket"]:
                    genre_counter.update({genre: 1})

            top_genre = genre_counter.most_common(1)

            if len(top_genre) != 0:
                cur_songs = song_mp_per_genre[top_genre[0][0]]
            else:
                cur_songs = song_mp
                
            ## modified for tag prediction
            tag_lst = self._tag_per_plylst(q, tag_per_song)
            tag_res = remove_seen(q["tags"], tag_lst)[:10]
            if len(tag_res) < 10:
                tag_res = remove_seen(q["tags"], tag_mp)[:10]
                
            ## modified for song prediction
            if len(q["songs"]) == 0:
                n_idx = idx + len(train)
                most_sim_lst = get_sim_plylst(tfidf_mat, given=n_idx, topn=30)
                cands = gather_cand(train, questions, most_sim_lst)
                song_res = remove_seen(q["songs"], cands)[:100]
                #print(n_idx, song_res)
            else:
                song_lst = self._cal_alsmodel(idx, num_users, usr_item_mat, als_model, voca_dict_t)  
                song_res = remove_seen(q["songs"], song_lst)[:100]
            
            if len(song_res) < 100:
                print('checked here', idx)
                song_res = remove_seen(q["songs"], cur_songs)[:100]

            answers.append({
                "id": q["id"],
                "songs": song_res,
                "tags": tag_res
            })

        return answers

    def run(self, song_meta_fname, train_fname, question_fname):
        print("Loading song meta...")
        song_meta_json = load_json(song_meta_fname)

        print("Loading train file...")
        train_data = load_json(train_fname)

        print("Loading question file...")
        questions = load_json(question_fname)

        print("Writing answers...")
        answers = self._generate_answers(song_meta_json, train_data, questions)
        write_json(answers, "results/results.json")
        
if __name__ == "__main__":
    fire.Fire(Als_with_CB)