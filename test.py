
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

def fun1():
    dataset_name =  "Musical_Instruments_5"
    save_path = os.path.join("saved", "{}_case.txt".format(dataset_name))

    test_dir = os.path.join("saved", dataset_name)

    model_names = ["CTR", "Narre", "NRT", "knowledge_copy", "gnn_copy", "gnn_knowledge", "gnn_knowledge_copy"]
    results = []
    all_datas = []
    refs_path = os.path.join(test_dir, "tests_" + model_names[0], "refs.txt")
    ratings_path = os.path.join(test_dir, "tests_" + model_names[0], "ratings.txt")
    with open(refs_path, 'r', encoding="utf-8") as f_ref, open(ratings_path, 'r', encoding="utf-8") as f_rat:
        ref_texts = f_ref.readlines()[:2000]
        ratings = f_rat.readlines()[:2000]
        datas = ["{:<5}\t{}".format(rat.strip(), ref.strip()) for rat, ref in zip(ratings, ref_texts)]
        all_datas.append(datas)

    for model_name in model_names:
        gens_path = os.path.join(test_dir, "tests_" + model_name, "gens.txt")
        predicting_path = os.path.join(test_dir, "tests_" + model_name, "predicting.txt")
        with open(gens_path, 'r', encoding="utf-8") as f_gen, open(predicting_path, 'r', encoding="utf-8") as f_pre:
            gen_texts = f_gen.readlines()[:2000]
            pre_ratings = f_pre.readlines()[:2000]
            datas = ["{:<5}\t{}".format(pre.strip()[:5], gen.strip()) for pre, gen in zip(pre_ratings, gen_texts)]
            all_datas.append(datas)

    columns = ["reference"] + model_names
    all_datas = np.array(all_datas).transpose(1, 0)
    with open(save_path, 'w', encoding="utf-8") as fout:
        for idx, datas in enumerate(all_datas):
            fout.write("###############  test instance {:04d}  ###############\n".format(idx))
            for column, data in zip(columns, datas):
                fout.write("{:<30}\t{}\n".format(column, data.strip()))
            fout.write('\n')
    pass

if __name__ == "__main__":
    fun1()