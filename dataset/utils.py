import json
import numpy as np
import os
from collections import defaultdict
from gensim.models import word2vec, Word2Vec
from tqdm import tqdm
import string
import numpy as np
import pickle
import random
import torch
import copy

def build_dict_from_txt(file, key_type=str, val_type=int, seg='\t'):
    ret_dict = {}
    with open(file, 'r', encoding="utf-8") as fin:
        for line in fin:
            key, val = line.strip("\n").split(seg)
            key, val = key_type(key), val_type(val)
            ret_dict[key] = val
    return ret_dict

def prepare_w2v(opt):
    word_dict = opt.get("dict")
    w2v_emb_size = opt["w2v_emb_size"]
    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "w2v_{}_{}.txt".format(len(word_dict), w2v_emb_size))
    save_npy_path = os.path.join(save_dir, "w2v_{}_{}.npy".format(len(word_dict), w2v_emb_size))
    opt["w2v_weight_path"] = save_npy_path

    if os.path.exists(save_npy_path):
        print("pretrained_word2vec {} is already exists".format(save_npy_path))
        return

    print("training word2vec weight from dataset {}".format(data_path))
    sentences = []
    if opt["data_source"] == "Amazon":
        text_fields = ["reviewText", "summary"]
    elif opt["data_source"] == "Yelp":
        text_fields = []
        #TODO:  add yelp
    with open(opt["data_path"]) as f_relations:
        for line in tqdm(f_relations, total=opt["total_instances"]):
            instance = json.loads(line)
            for field in text_fields:
                text = instance[field]
                sentences.append(list(word_dict.tokenize(text)))

    model = word2vec.Word2Vec(sentences, size=w2v_emb_size, window=5, min_count=0, workers=4)
    model.wv.save_word2vec_format(save_path, binary=False)
    to_save = np.random.randn(len(word_dict), w2v_emb_size)
    for vec, tok in zip(model.wv.vectors, model.wv.index2word):
        idx = word_dict.tok2ind.get(tok, None)
        if idx is not None:
            to_save[idx, :] = vec
    np.save(save_npy_path, to_save)

def build_conceptnet(opt):
    '''
    save:
        tok2uri.txt
        indv2indc.txt: vocab index to conceptnet index
        indv2idnc.txt: exchange vocab index to conceptnet index
        request_cache.txt: cache of request
        conceptnet_emb.npy: numpy weight   N * 300,  N is concepnet vocab len
        conceptnet_neibors_emb.npy: numpy weight N * words_topk.
    '''
    word_dict = opt.get("dict")
    save_dir = opt["conceptnet_dir"]
    ind2tok = word_dict.ind2tok
    tok2ind = word_dict.tok2ind

    tok2uri = None
    uri2indc = None
    conceptnet_emb = None
    ind2topk_emb = None

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    this_save_dir = os.path.join(save_dir, data_name)
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)

    numberbatch_path = os.path.join(save_dir, "numberbatch-en.txt")
    if not os.path.exists(numberbatch_path):
        raise("please Download concept numberbatch at https://github.com/commonsense/conceptnet-numberbatch")
    tok2uri_path = os.path.join(this_save_dir, "tok2uri.txt")
    uri2indc_path = os.path.join(this_save_dir, "uri2indc.txt")
    indv2idnc_path = os.path.join(this_save_dir, "indv2indc.txt")
    request_cache_path = os.path.join(save_dir, "request_cache.txt")
    conceptnet_emb_path = os.path.join(save_dir, "conceptnet_emb_{}.npy".format(opt["conceptnet_emb_type"]))
    conceptnet_neibors_emb_path = os.path.join(save_dir, "conceptnet_{}neibors_emb.npy".format(opt["words_topk"]))
    opt["conceptnet_emb_path"] = conceptnet_emb_path
    opt["conceptnet_neibors_emb_path"] = conceptnet_neibors_emb_path
    opt["ind2topk_conceptnet_ind_path"] = indv2idnc_path

    with open(numberbatch_path) as f_numberbatch:
        for line in f_numberbatch:
            conceptnet_len, conceptnet_emb_size = line.split()
            conceptnet_len, conceptnet_emb_size = int(conceptnet_len), int(conceptnet_emb_size)
            opt["conceptnet_len"] = conceptnet_len
            opt["conceptnet_emb_size"] = conceptnet_emb_size
            break

    ###############Create tok2uri################
    if os.path.exists(tok2uri_path):
        print("tok2uri file {} is already exists".format(tok2uri_path))
    else:
        print("Creating tok2uri.txt, saving at {}".format(tok2uri_path))

        def tok2uri_function(tok, lanuage='en'):
            from .text2url import standardized_uri
            return standardized_uri(lanuage, tok)

        tok2uri = {}
        with open(tok2uri_path, 'w', encoding="utf-8") as f_tok2uri:
            for tok in tqdm(ind2tok.values()):
                uri = tok2uri_function(tok)
                uri = uri.split('/')[-1]
                f_tok2uri.write("{}\t{}\n".format(tok, uri))
                tok2uri[tok] = uri


    ###############Create uri2indc################
    if os.path.exists(uri2indc_path):
        print("uri2indc file {} is already exists".format(uri2indc_path))
        uri2indc = build_dict_from_txt(uri2indc_path, key_type=str, val_type=int)
    else:
        print("Creating uri2indc.txt, saving at {}".format(uri2indc_path))
        uri2indc = {}
        with open(numberbatch_path, 'r', encoding="utf-8") as f_numberbatch, open(uri2indc_path, 'w', encoding="utf-8") as f_uri2indc:
            for i, line in enumerate(tqdm(f_numberbatch, total=conceptnet_len+1)):
                if i != 0:
                    uri = line.split()[0]
                    f_uri2indc.write("{}\t{}\n".format(uri, i-1))
                    uri2indc[uri] = i-1


    ###############Create indv2indc################
    if os.path.exists(indv2idnc_path):
        print("indv2indc file {} is already exists".format(indv2idnc_path))
        indv2indc = build_dict_from_txt(indv2idnc_path, key_type=int, val_type=int)
    else:
        print("Creating indv2indc, saving at {}".format(indv2idnc_path))
        indv2indc = {}
        if os.path.exists(request_cache_path):
            request_cache = build_dict_from_txt(request_cache_path, key_type=str, val_type=int)
        else:
            request_cache = {}
        len_cache = len(request_cache)
        def find_uri_indc(uri):
            if uri in uri2indc:
                return uri2indc[uri]
            elif uri in request_cache:
                return request_cache[uri]
            else:
                import requests
                related_url = "http://api.conceptnet.io/related/c/en/{}?filter=/c/en"
                while True:
                    try:
                        obj = requests.get(related_url.format(uri)).json()
                        break
                    except Exception as e:
                        print(e)
                        continue
                indc = None
                for related in obj["related"]:
                    related_uri = related["@id"].split('/')[-1]
                    if related_uri in uri2indc:
                        indc =  uri2indc[related_uri]
                        break
                indc = uri2indc[tok2uri[word_dict.null_tok]] if indc is None else indc
                request_cache[uri] = indc
            return indc

        with open(indv2idnc_path, 'w', encoding="utf-8") as f_indv2indc:
            if tok2uri is None:
                tok2uri = build_dict_from_txt(tok2uri_path, key_type=str, val_type=str)
            if uri2indc is None:
                uri2indc = build_dict_from_txt(uri2indc_path, key_type=str, val_type=int)
            none_uris = []

            for indv, tok in tqdm(ind2tok.items()):
                indc = find_uri_indc(tok2uri[tok])
                f_indv2indc.write("{}\t{}\n".format(indv, indc))
                indv2indc[indv] = indc
                if indc == uri2indc[tok2uri[word_dict.null_tok]]:
                    none_uris.append(tok)
        #保存请求缓存
        if len(request_cache) > len_cache:
            with open(request_cache_path, 'w', encoding="utf-8") as f_cache:
                for uri, indc in request_cache.items():
                    f_cache.write("{}\t{}\n".format(uri, indc))
    opt["indv2indc"] = indv2indc

    ###############Create conceptnet_emb and neibors_emb################
    if os.path.exists(conceptnet_emb_path):
        print("Conceptnet embedding {} is already exists".format(conceptnet_emb_path))
    else:
        print("Creating conceptnet_emb, saving at {}".format(conceptnet_emb_path))
        import pandas as pd
        with open(numberbatch_path, 'r', encoding="utf-8") as f_numberbatch:
            for line in f_numberbatch:
                break
            emb_df = pd.read_table(f_numberbatch, sep=' ', header=None, index_col=0).astype(getattr(np, opt["conceptnet_emb_type"]))
            np.save(conceptnet_emb_path, emb_df.values)

    if os.path.exists(conceptnet_neibors_emb_path):
        print("Conceptnet neighbors' embedding {} is already exists".format(conceptnet_neibors_emb_path))
    else:
        print("Creating conceptnet_neighbor_emb, saving at {}".format(conceptnet_neibors_emb_path))
        import torch
        try:
            embeddings = emb_df.values
        except Exception:
            embeddings = np.load(conceptnet_emb_path)
        embeddings = torch.from_numpy(embeddings).to(opt["device"])
        neighbors_emb = torch.LongTensor(embeddings.size(0), opt["words_topk"]).to(opt["device"])
        batch_size = 1024
        shift = 0
        for i in tqdm(range(embeddings.size(0) // batch_size + 1)):
            emb = embeddings[shift : shift + batch_size, :]
            length = emb.size(0)
            match = torch.matmul(emb, embeddings.t())
            neighbors_emb[shift:shift+length, :] = torch.topk(match, k=opt["words_topk"], dim=-1).indices
            shift += batch_size
        neighbors_emb = neighbors_emb.cpu().numpy()
        np.save(conceptnet_neibors_emb_path, neighbors_emb)

def build_user_item_matrix(opt):
    '''
    save:
        user2ind.txt
        item2ind.txt
        user_item_matrix.json
    '''
    word_dict = opt.get("dict")
    data_path = opt["data_path"]
    total_instances = opt["total_instances"]
    min_tip_len = opt["min_tip_len"]
    save_dir = opt["save_dir"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    user2ind_path = os.path.join(save_dir, "user2ind.txt")
    item2ind_path = os.path.join(save_dir, "item2ind.txt")
    user_item_matrix_path = os.path.join(save_dir, "user_item_matrix.json")
    opt["user_item_matrix_path"] = user_item_matrix_path

    if opt["data_source"] == "Amazon":
        text_field = "reviewText"
        user_field = "reviewerID"
        item_field = "asin"
        rating_field = "overall"
        tip_field = "summary"
    elif opt["data_source"] == "Yelp":
        text_field = ""
        user_field = ""
        item_field = ""
        rating_field = ""
        #TODO:  add yelp

    user_item_matrix = {}

    #build user2ind and item2ind
    user2ind = {}
    item2ind = {}
    if os.path.exists(user2ind_path) and os.path.exists(item2ind_path):
        print("user2ind {} and item2ind file {} is already exist".format(user2ind_path, item2ind_path))
        user2ind = build_dict_from_txt(user2ind_path, key_type=str, val_type=int)
        item2ind = build_dict_from_txt(item2ind_path, key_type=str, val_type=int)
    else:
        print("Building user2ind {} and item2ind file {} ".format(user2ind_path, item2ind_path))
        with open(data_path, 'r', encoding="utf-8") as f_relations:
            for line in tqdm(f_relations, total=total_instances):
                instance = json.loads(line)
                user = instance[user_field]
                item = instance[item_field]
                if user not in user2ind:
                    user2ind[user] = len(user2ind)
                if item not in item2ind:
                    item2ind[item] = len(item2ind)
        with open(user2ind_path, 'w', encoding="utf-8") as f_user2ind, open(item2ind_path, 'w', encoding="utf-8") as f_item2ind:
            key = lambda x:x[1]
            for (user, uind) in sorted(user2ind.items(), key=key):
                f_user2ind.write("{}\t{}\n".format(user, uind))
            for (item, iind) in sorted(item2ind.items(), key=key):
                f_item2ind.write("{}\t{}\n".format(item, iind))

    opt["user2ind"] = user2ind
    opt["item2ind"] = item2ind
    opt["num_users"] = len(user2ind)
    opt["num_items"] = len(item2ind)

    #write user_item_matrix.json
    if os.path.exists(user_item_matrix_path):
        print("User and item interactive matrix file {} is already exist".format(user_item_matrix_path))
    else:
        print("Building user and item interactive matrix file {}.".format(user_item_matrix_path))
        with open(data_path, 'r', encoding="utf-8") as f_relations:
            with open(user_item_matrix_path, 'w', encoding="utf-8") as f_matrix:
                for line in tqdm(f_relations, total=total_instances):
                    instance = json.loads(line)
                    user = instance[user_field]
                    item = instance[item_field]
                    rating = float(instance[rating_field])
                    if user in user_item_matrix:
                        if item in user_item_matrix[user]:
                            user_item_matrix[user][item].append(rating)
                        else:
                            user_item_matrix[user][item] = [rating]
                    else:
                        user_item_matrix[user] = {item:[rating]}
                user_item_matrix = sorted(user_item_matrix.items(), key=lambda x:user2ind[x[0]])
                for user, itemdict in user_item_matrix:
                    for item, rating in itemdict.items():
                        itemdict[item] = sum(rating) / len(rating)
                    line = itemdict
                    f_matrix.write(json.dumps(line) + '\n')

def build_datasets(opt):
    '''
    to save:    train.pkl, val.pkl, test.pkl
        users: list[int] users' ind
        items: list[int] items' ind
        ratings: list[float] ratings of item from user
        tips: str tips of item from user
    '''
    word_dict = opt.get("dict")
    toolkits = opt["toolkits"]
    data_path = opt["data_path"]
    total_instances = opt["total_instances"]
    min_tip_len = opt["min_tip_len"]
    splits = opt["splits"]
    save_dir = opt["save_dir"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name, "dataset_min_tip_len_{}".format(min_tip_len))
    dataset_path = os.path.join(save_dir, "dataset.pkl")

    opt["dataset_path"] = dataset_path

    user2ind = opt["user2ind"]
    item2ind = opt["item2ind"]
    if os.path.exists(save_dir):
        print("dataset files is already exist in {}".format(save_dir))
    else:
        os.makedirs(save_dir)
        print("Building dataset files save in {}".format(save_dir))
        if opt["data_source"] == "Amazon":
            text_field = "reviewText"
            user_field = "reviewerID"
            item_field = "asin"
            rating_field = "overall"
            tip_field = "summary"
            time_field = "unixReviewTime"
        elif opt["data_source"] == "Yelp":
            text_field = ""
            user_field = ""
            item_field = ""
            rating_field = ""
            #TODO:  add yelp


        with open(data_path, 'r', encoding="utf-8") as f_instances:
            """
                test samples' time can not earlier than train samples
            """
            def filter(all_instances):
                ret = []
                for line in all_instances:
                    instance = json.loads(line)
                    text = instance[text_field]
                    rating = float(instance[rating_field])
                    tip = instance[tip_field]
                    time_stamp = instance[time_field]

                    if (len(tip.split()) < min_tip_len):
                        for tip in word_dict.sent_tok.tokenize(text):
                            if (len(tip.split()) >= min_tip_len):
                                break
                    if (len(tip.split()) < min_tip_len):
                        continue
                    instance[tip_field] = tip
                    ret.append(instance)
                return ret

            all_instances = f_instances.readlines()
            all_instances = filter(all_instances)
            timestamps = sorted([instance[time_field] for instance in all_instances])
            # all_instances.sort(key=lambda x:x[time_field])
            # random.shuffle(all_instances)
            splits_list = list(map(float, splits.strip('\n').split(':')))
            train_p, val_p, test_p = list(map(lambda x:x / sum(splits_list), splits_list))
            split1 = timestamps[int(len(all_instances) * train_p)]
            split2 = timestamps[int(len(all_instances) * (train_p + val_p))]

            train_raw = []
            val_raw = []
            test_raw = []
            train_uind = {}
            train_iind = {}
            for instance in tqdm(all_instances):
                user_ind = user2ind[instance[user_field]]
                item_ind = item2ind[instance[item_field]]
                text = instance[text_field]
                rating = float(instance[rating_field])
                tip = instance[tip_field]
                time_stamp = instance[time_field]
                if time_stamp < split1:
                    train_raw.append((user_ind, item_ind, rating, tip, text))
                    train_uind[user_ind] = 1
                    train_iind[item_ind] = 1
                elif time_stamp < split2:
                    if user_ind in train_uind and item_ind in train_iind:
                        val_raw.append((user_ind, item_ind, rating, tip, text))
                elif user_ind in train_uind and item_ind in train_iind:
                    test_raw.append((user_ind, item_ind, rating, tip, text))
            #sort by item ind
            train_raw.sort(key=lambda x:x[1])
            val_raw.sort(key=lambda x:x[1])
            test_raw.sort(key=lambda x:x[1])
            with open(dataset_path, 'wb') as f_save:
                pickle.dump([train_raw, val_raw, test_raw], f_save)

def build_relation_datas(opt):
    '''
    to save:
        relations:  in relations.pkl
            user_ne_items:  defaultdict(list) key: user_ind, val: user's neighbor items' ind
            item_ne_users:  defaultdict(list) key: item_ind, val: item's neighbor users' ind
            user_ne_users:  defaultdict(list) key: user_ind, val: user's neighbor users' ind
            item_ne_items:  defaultdict(list) key: item_ind, val: item's neighbor items' ind
            user_item_review:  defaultdict(str) key: (user_ind, item_ind), val: review of item by user
            pair2ind:  defaultdict(int) key: (user_ind, item_ind), val: review's ind of item by user
    '''
    word_dict = opt.get("dict")
    data_path = opt["data_path"]
    dataset_path = opt["dataset_path"]
    total_instances = opt["total_instances"]
    save_dir = opt["save_dir"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    user_ne_items_path = os.path.join(save_dir, "user_ne_items.txt")
    item_ne_users_path = os.path.join(save_dir, "item_ne_users.txt")
    user_item_review_path = os.path.join(save_dir, "user_item_review.txt")
    pair2ind_path = os.path.join(save_dir, "pair2ind.txt")
    relations_path = os.path.join(save_dir, "relations.pkl")
    opt["relations_path"] = relations_path
    user2ind = opt["user2ind"]
    item2ind = opt["item2ind"]

    if opt["data_source"] == "Amazon":
        text_field = "reviewText"
        user_field = "reviewerID"
        item_field = "asin"
        rating_field = "overall"
        tip_field = "summary"
    elif opt["data_source"] == "Yelp":
        text_field = ""
        user_field = ""
        item_field = ""
        rating_field = ""
        #TODO:  add yelp
    user_ne_items = defaultdict(list)
    item_ne_users = defaultdict(list)
    user_item_tip = defaultdict(str)
    user_item_review = defaultdict(str)
    pair2ind = defaultdict(int)
    if os.path.exists(relations_path):
        print("relations file {} is already exist".format(relations_path))
    else:
        print("Building relations file {}.".format(relations_path))
        with open(dataset_path, 'rb') as fd:
            train_raw, _, _ = pickle.load(fd)
            for user_ind, item_ind, _, tip, text in train_raw:
                pair = (user_ind, item_ind)
                user_item_tip[pair] += tip
                user_item_review[pair] += text
                if item_ind not in user_ne_items[user_ind]:
                    user_ne_items[user_ind].append(item_ind)
                if user_ind not in item_ne_users[item_ind]:
                    item_ne_users[item_ind].append(user_ind)
                if pair not in pair2ind:
                    pair2ind[pair] = len(pair2ind)
        user_ne_users = get_neighbor_users(opt)
        item_ne_items = get_neighbor_items(opt)

        relations = (user_ne_items, item_ne_users, user_ne_users, item_ne_items, user_item_tip, user_item_review, pair2ind)
        with open(relations_path, 'wb') as f_relations:
            pickle.dump(relations, f_relations)

        with open(user_item_review_path, 'w', encoding="utf-8") as f_uir:
            for (user, item), review in user_item_review.items():
                f_uir.write("{}\t{}\t{}\n".format(user, item, review))

        with open(pair2ind_path, 'w', encoding="utf-8") as f_p2i:
            for (user, item), ind in pair2ind.items():
                f_p2i.write("{}\t{}\t{}\n".format(user, item, ind))

        with open(user_ne_items_path, 'w', encoding="utf-8") as f_ui:
            for user, items in user_ne_items.items():
                f_ui.write("{}\t{}\n".format(user, " ".join(map(str, items))))
        with open(item_ne_users_path, 'w', encoding="utf-8") as f_iu:
            for item, users in item_ne_users.items():
                f_iu.write("{}\t{}\n".format(item, "".join(map(str, users))))

def prepaire_graph_info(opt):
    '''
    to save:
        graph_info:  in graph_info.pkl
            user2text_vectors:  defaultdict(list) key: userind, val: vectors of user text
            item2text_vectors:  defaultdict(list) key: itemind, val: vectors of item text
            review2text_vectors:  defaultdict(list) key: (uind, iind), val: vectors of review text of item from user
    '''
    word_dict = opt.get("dict")
    toolkits = opt["toolkits"]
    data_path = opt["data_path"]
    total_instances = opt["total_instances"]
    save_dir = opt["save_dir"]
    user2ind = opt["user2ind"]
    item2ind = opt["item2ind"]
    max_neighbors = opt["max_neighbors"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    graph_info_path = os.path.join(save_dir, "graph_info.pkl")
    opt["graph_info_path"] = graph_info_path
    with open(opt["relations_path"], "rb") as f_relations:
        user_ne_items, item_ne_users, user_ne_users, item_ne_items, \
                user_item_tip, user_item_review, pair2ind = pickle.load(f_relations)

    def get_user_text(uind, user_ne_items, user_item_review):
        ne_items = user_ne_items[uind][:max_neighbors]
        ret = ""
        for iind in ne_items:
            edge = (uind, iind)
            ret += user_item_review[edge].rstrip('\n')
        return ret

    def get_item_text(iind, item_ne_users, user_item_review):
        ne_users = item_ne_users[iind][:max_neighbors]
        ret = ""
        for uind in ne_users:
            edge = (uind, iind)
            ret += user_item_review[edge].rstrip('\n')
        return ret

    user2text_vectors = defaultdict(list)
    item2text_vectors = defaultdict(list)
    tip2text_vectors = defaultdict(list)
    review2text_vectors = defaultdict(list)
    if os.path.exists(graph_info_path):
        print("graph info file {} is already exist".format(graph_info_path))
    else:
        print("Building graph info file {}.".format(graph_info_path))
        print("Building reviews graph info")
        for (uind, iind), tip in tqdm(user_item_tip.items()):
            tip2text_vectors[(uind, iind)] = toolkits.text2vectors(tip)
        for (uind, iind), review in tqdm(user_item_review.items()):
            review2text_vectors[(uind, iind)] = toolkits.text2vectors(review)
        print("Building users graph info")
        for uind in tqdm(user2ind.values()):
            user_text = get_user_text(uind, user_ne_items, user_item_review)
            user2text_vectors[uind] = toolkits.text2vectors(user_text)
        print("Building items graph info")
        for iind in tqdm(item2ind.values()):
            item_text = get_item_text(iind, item_ne_users, user_item_review)
            item2text_vectors[iind] = toolkits.text2vectors(item_text)

        graph_infos = (user2text_vectors, item2text_vectors, tip2text_vectors, review2text_vectors)
        with open(graph_info_path, 'wb') as f_ginfo:
            pickle.dump(graph_infos, f_ginfo)

def get_neighbor_users(opt):
    '''
    save:
        user_ne_k.txt
    '''
    topk = opt["user_topk"]
    user_item_matrix_path = opt["user_item_matrix_path"]
    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    user_ne_k_path = os.path.join(save_dir, "user_ne_{}.txt".format(topk))

    user_ne_users = defaultdict(list)
    if os.path.exists(user_ne_k_path):
        print("The file of user-user neighbors is Already exists in {}".format(user_ne_k_path))
        with open(user_ne_k_path, 'r', encoding="utf-8") as f_user_ne:
            for line in f_user_ne:
                user, ne_users = line.split('\t')
                user = int(user)
                ne_users = list(map(int, ne_users.split()))
                user_ne_users[user] = ne_users
    else:
        print("Building the file of user-user neighbors {}".format(user_ne_k_path))
        from datasketch import MinHashLSHForest, MinHash
        minHashs = []
        forset = MinHashLSHForest(num_perm=128)
        with open(user_item_matrix_path, 'r', encoding="utf-8") as f_user_item:
            for i, line in tqdm(enumerate(f_user_item), total=opt["num_users"]):
                line = json.loads(line)
                m = MinHash(num_perm=128)
                for d in line.keys():
                    m.update(d.encode("utf8"))
                forset.add(i, m)
                minHashs.append(m)
        forset.index()

        with open(user_ne_k_path, 'w', encoding="utf-8") as f_user_ne:
            for h, minHash in tqdm(enumerate(minHashs), total=opt["num_users"]):
                topk_relate = forset.query(minHash, 10)
                topk_relate.remove(h)
                if len(topk_relate) > 0:
                    line_ne = "{}\t{}\n".format(h, ' '.join(map(str, topk_relate)))
                    user_ne_users[h] = topk_relate
                    f_user_ne.write(line_ne)
    return user_ne_users

def get_neighbor_items(opt):
    '''
    save:
        item_ne.txt
    '''
    min_support = opt["min_support"]
    min_conf = opt["min_conf"]
    item2ind = opt["item2ind"]
    user_item_matrix_path = opt["user_item_matrix_path"]
    # ****************
    # item2ind = build_dict_from_txt("saved/Electronics_5/item2ind.txt", key_type=str, val_type=int)
    # user2ind = build_dict_from_txt("saved/Electronics_5/user2ind.txt", key_type=str, val_type=int)
    # user_item_matrix_path = "saved/Electronics_5/user_item_matrix.json"
    # opt["num_users"] = len(user2ind)
    #*****************
    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    item_ne_path = os.path.join(save_dir, "item_ne.txt")

    item_ne_items = defaultdict(list)
    if os.path.exists(item_ne_path):
        print("The file of item-item neighbor is Already exists in {}".format(item_ne_path))
        with open(item_ne_path, 'r', encoding="utf-8") as f_item_ne:
            for line in f_item_ne:
                item, items_ne = line.split('\t')
                item = int(item)
                items_ne = list(map(lambda x:int(x.split(':')[0]), items_ne.split()))
                item_ne_items[item] = items_ne
    else:
        print("Building the file of item-item neighbors is Already exists in {}".format(item_ne_path))

        support1 = defaultdict(float)
        support2 = defaultdict(float)
        min_sum = opt["num_users"] * min_support
        with open(user_item_matrix_path, 'r', encoding="utf-8") as f_user_item:
            for i, line in tqdm(enumerate(f_user_item), total=opt["num_users"]):
                line = json.loads(line)
                items = list(map(lambda x:item2ind[x], line.keys()))
                for i in range(len(items)):
                    support1[str(items[i])] += 1.
        for key, val in list(support1.items()):
            if val < min_sum:
                del support1[key]

        with open(user_item_matrix_path, 'r', encoding="utf-8") as f_user_item:
            for i, line in tqdm(enumerate(f_user_item), total=opt["num_users"]):
                line = json.loads(line)
                items = list(map(lambda x:item2ind[x], line.keys()))
                for i in range(len(items)):
                    if str(items[i]) not in support1:
                        continue
                    for j in range(i+1, len(items)):
                        if str(items[j]) not in support1:
                            continue
                        h, t = items[i], items[j]
                        pair = (h, t) if (h, t) in support2 else (t, h)
                        support2[pair] += 1.
        for key, val in tqdm(list(support2.items())):
            if val < min_sum:
                del support2[key]

        confs = defaultdict(dict)
        for (i, j), val in support2.items():
            confs_i_j = support2[(i, j)] / support1[str(i)]
            confs_j_i = support2[(i, j)] / support1[str(j)]
            if confs_i_j >= min_conf:
                confs[str(i)][str(j)] = confs_i_j
            if confs_j_i >= min_conf:
                confs[str(j)][str(i)] = confs_j_i
        del support1, support2

        with open(item_ne_path, 'w', encoding="utf-8") as f_item_ne:
            for h, val in confs.items():
                tails = []
                c_vals = []
                for t, c_val in val.items():
                    tails.append(t)
                    c_vals.append(c_val)
                item_ne_items[h] = tails
                line_ne = "{}\t{}\n".format(h, ' '.join(("{}:{}".format(i, v) for i, v in zip(tails, c_vals))))
                f_item_ne.write(line_ne)
    return item_ne_items

def prepare_datas(opt):
    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    toolkits_path = os.path.join(save_dir, "toolkits.pkl")

    prepare_w2v(opt)
    build_conceptnet(opt)
    build_user_item_matrix(opt)
    toolkits = Toolkits(opt)
    opt["toolkits"] = toolkits
    build_datasets(opt)
    build_relation_datas(opt)
    prepaire_graph_info(opt)
    if not os.path.exists(toolkits_path):
        with open(os.path.join(save_dir, "toolkits.pkl"), 'wb') as fb:
            pickle.dump(toolkits, fb)

class Toolkits(object):
    def __init__(self, opt):
        self.word_dict = opt["dict"]
        self.user2ind_dict = opt["user2ind"]
        self.item2ind_dict = opt["item2ind"]
        self.indv2indc_dict = opt["indv2indc"]
        self.max_text_len = opt["max_text_len"]
        self.max_sent_len = opt["max_sent_len"]
        self.device = opt["device"]
        self.null_ind = self.tok2ind(self.word_dict.null_tok)
        self.unk_ind = self.tok2ind(self.word_dict.unk_tok)
        self.start_ind = self.tok2ind(self.word_dict.start_tok)
        self.end_ind = self.tok2ind(self.word_dict.end_tok)

    def tok2ind(self, tok):
        return self.word_dict.tok2ind.get(tok, self.word_dict.tok2ind[self.word_dict.unk_tok])

    def ind2tok(self, ind):
        return self.word_dict.ind2tok.get(ind)

    def user2ind(self, user):
        return self.user2ind_dict.get(user, -1)

    def item2ind(self, item):
        return self.item2ind_dict.get(item, -1)

    def indv2indc(self, indv):
        return self.indv2indc_dict.get(indv, self.indv2indc_dict[0])

    def uind2entity(self, uind):
        return uind

    def iind2entity(self, iind):
        return len(self.user2ind_dict) + iind

    def indvvec2indcvec(self, indvvec):
        ret = []
        for indv in indvvec:
            ret.append(self.indv2indc(indv))
        return ret

    def text2vec(self, text, add_start=False, add_end=False, save_stop_words=True):
        ret = []
        if add_start:
            ret.append(self.tok2ind(self.word_dict.start_tok))
        # translator = str.maketrans('', '', string.punctuation)
        # text = text.translate(translator)
        for tok in self.word_dict.tokenize(text, save_stop_words=save_stop_words)[:self.max_text_len]:
            ind = self.tok2ind(tok)
            if ind != self.unk_ind:
                ret.append(self.tok2ind(tok))
        if add_end:
            ret.append(self.tok2ind(self.word_dict.end_tok))
        if len(ret) == 0:
            ret.append(self.tok2ind(self.word_dict.null_tok))
        return ret[:self.max_text_len]

    def vecs2texts(self, vecs):
        rets = []
        for vec in vecs:
            ret = ""
            for ind in vec:
                if isinstance(ind, torch.Tensor):
                    ind = ind.item()
                tok = self.ind2tok(ind)
                if tok == self.word_dict.end_tok:
                    break
                ret += (tok + " ")
            rets.append(ret + '\n')
        return rets

    def text2sentences_vec(self, text):
        ret = []
        max_len = 0
        sentences = self.word_dict.sent_tok.tokenize(text)[:self.max_sent_len]
        for sent in sentences:
            ret.append(self.text2vec(sent))
            max_len = max(max_len, len(ret[-1]))
        if len(ret) == 0:
            ret.append([self.tok2ind(self.word_dict.null_tok)])
            max_len = 1
        return ret, max_len

    def pad2d(self, datas, pad_len, pad_ind=None):
        pad_data = copy.deepcopy(datas)
        pad_ind = pad_ind if pad_ind is not None else self.tok2ind(self.word_dict.null_tok)
        for data in pad_data:
            if pad_len > len(data):
                data.extend([pad_ind] * (pad_len - len(data)))
            else:
                del data[pad_len:]
        return np.array(pad_data)

    def pad3d(self, datas, len1, len2, pad_ind=None):
        pad_datas = copy.deepcopy(datas)
        pad_ind = pad_ind if pad_ind is not None else self.tok2ind(self.word_dict.null_tok)
        pad_dim1 = [pad_ind] * len2
        for i, data in enumerate(pad_datas):
            pad_data = self.pad2d(data, len2)
            if len1 > len(data):
                pad_data = np.pad(pad_data, ((0, len1 - len(data)), (0, 0)), constant_values=(pad_ind,))
            else:
                pad_data = pad_data[:len1]
            pad_datas[i] = pad_data
        return np.array(pad_datas)

    def texts2vec(self, texts, max_copy_len):
        text_vecs = [self.text2vec(text)[:max_copy_len] for text in texts]
        text_lens = torch.Tensor([len(text) for text in text_vecs])
        text_pad_vecs = self.pad2d(text_vecs, max_copy_len)
        return torch.from_numpy(text_pad_vecs).long(), text_lens

    def text2vectors(self, text):
        text_vec = self.text2vec(text)[:self.max_text_len]
        sents_vec, maxlen_of_every_sent = self.text2sentences_vec(text)
        conceptnet_text_vec = self.indvvec2indcvec(text_vec)
        return [text_vec, sents_vec, conceptnet_text_vec, maxlen_of_every_sent]

    def batch_vectors(self, vectors, sorted=True):
        text_vecs = [vector[0] for vector in vectors]
        sents_vecs = [vector[1] for vector in vectors]
        conceptnet_text_vecs = [vector[2] for vector in vectors]
        every_sent_pad_lens = [vector[3] for vector in vectors]

        text_lens = np.array([len(text) for text in text_vecs])
        sents_lens = np.array([len(sent) for sent in sents_vecs])

        text_pad_len = min(max(text_lens), self.max_text_len)
        sents_pad_len = min(max(sents_lens), self.max_sent_len)
        every_sent_pad_len = min(max(every_sent_pad_lens), self.max_text_len)
        text_pad_vecs = self.pad2d(text_vecs, text_pad_len)
        sents_pad_vecs = self.pad3d(sents_vecs, sents_pad_len, every_sent_pad_len)
        conceptnet_text_pad_vecs = self.pad2d(conceptnet_text_vecs, text_pad_len)

        text_lens[text_lens > self.max_text_len] = self.max_text_len
        sents_lens[sents_lens > self.max_sent_len] = self.max_sent_len

        # reset_sorted_idx = np.array([])
        # if sorted:
        #     text_sorted_idx = np.argsort(text_lens)[::-1]
        #     sents_sorted_idx = np.argsort(sents_lens)[::-1]
        #     text_lens = text_lens[text_sorted_idx]
        #     sents_lens = sents_lens[sents_sorted_idx]
        #     text_pad_vecs = text_pad_vecs[text_sorted_idx]
        #     sents_pad_vecs = sents_pad_vecs[sents_sorted_idx]
        #     conceptnet_text_pad_vecs = conceptnet_text_pad_vecs[text_sorted_idx]
        #     reset_sorted_idx = np.argsort(text_sorted_idx)

        return (torch.from_numpy(text_pad_vecs).long().to(self.device), torch.from_numpy(text_lens).long(), \
                torch.from_numpy(sents_pad_vecs).long().to(self.device), torch.from_numpy(sents_lens).long(), \
                    torch.from_numpy(conceptnet_text_pad_vecs).long().to(self.device))

    def texts2vectors(self, texts, sorted=True):
        text_vecs = []
        sents_vecs = []
        conceptnet_text_vecs = []
        every_sent_pad_lens = []
        for text in texts:
            text_vec = self.text2vec(text)
            sents_vec, maxlen_of_every_sent = self.text2sentences_vec(text)
            conceptnet_text_vec = self.indvvec2indcvec(text_vec)
            text_vecs.append(text_vec)
            sents_vecs.append(sents_vec)
            conceptnet_text_vecs.append(conceptnet_text_vec)
            every_sent_pad_lens.append(maxlen_of_every_sent)

        text_lens = np.array([len(text) for text in text_vecs])
        sents_lens = np.array([len(sent) for sent in sents_vecs])

        text_pad_len = min(max(text_lens), self.max_text_len)
        sents_pad_len = min(max(sents_lens), self.max_sent_len)
        every_sent_pad_len = min(max(every_sent_pad_lens), self.max_text_len)
        text_pad_vecs = self.pad2d(text_vecs, text_pad_len)
        sents_pad_vecs = self.pad3d(sents_vecs, sents_pad_len, every_sent_pad_len)
        conceptnet_text_pad_vecs = self.pad2d(conceptnet_text_vecs, text_pad_len)

        text_lens[text_lens > self.max_text_len] = self.max_text_len
        sents_lens[sents_lens > self.max_sent_len] = self.max_sent_len

        # reset_sorted_idx = np.array([])
        # if sorted:
        #     text_sorted_idx = np.argsort(text_lens)[::-1]
        #     sents_sorted_idx = np.argsort(sents_lens)[::-1]
        #     text_lens = text_lens[text_sorted_idx]
        #     sents_lens = sents_lens[sents_sorted_idx]
        #     text_pad_vecs = text_pad_vecs[text_sorted_idx]
        #     sents_pad_vecs = sents_pad_vecs[sents_sorted_idx]
        #     conceptnet_text_pad_vecs = conceptnet_text_pad_vecs[text_sorted_idx]
        #     reset_sorted_idx = np.argsort(text_sorted_idx)

        return (torch.from_numpy(text_pad_vecs).long().to(self.device), torch.from_numpy(text_lens).long(), \
                torch.from_numpy(sents_pad_vecs).long().to(self.device), torch.from_numpy(sents_lens).long(), \
                    torch.from_numpy(conceptnet_text_pad_vecs).long().to(self.device))
