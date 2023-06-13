import argparse
from dataset.build_dict import Dict
from dataset import utils
from dataset.DataManager import RecDatasetManager
from modules.test_seq2seq import Seq2Seq
import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import time
import os
import pickle
import copy


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def prepare_datas(opt):
    word_dict = Dict(opt)
    word_dict.build(sort=True)
    opt["dict"] = word_dict
    utils.prepare_datas(opt)

def train(model, optimizer, epoch, data_loader, metrics, writer, print_step, toolkits):    
    s = time.time()
    model.train()
    global_step = metrics["global_step"]
    print_step = print_step

    for i, (uinds, iinds, ratings, tips_vec) in enumerate(data_loader):
        optimizer.zero_grad()
        expalin_scores, explain_hat = model(uinds, iinds, tips_vec)
        loss_per_tok, nll_loss, target_tokens, correct = model.loss_gen(expalin_scores, explain_hat, tips_vec)
        loss_per_tok.backward()
        optimizer.step()
        metrics["train_loss_gen"] += loss_per_tok.item()
        metrics["train_tok_acc"] += (correct / target_tokens)
        if (global_step + i + 1) % print_step == 0:
            tips = toolkits.vecs2texts(tips_vec)
            predicts = toolkits.vecs2texts(explain_hat)
            for tip, predict in zip(tips, predicts):
                print("model: {}system: {}".format(tip, predict))

            metrics["train_loss_gen"] /= print_step
            metrics["train_tok_acc"] /= print_step
            if metrics["train_loss_gen"] < metrics["min_train_loss"]:
                metrics["min_train_loss"] = metrics["train_loss_gen"]
            e = time.time()
            
            writer.add_scalar("train/train_loss_gen", metrics["train_loss_gen"], global_step=global_step + i)
            writer.add_scalar("train/train_tok_acc", metrics["train_tok_acc"], global_step=global_step + i)

            print("train [{:02d}, {:06d}] time_use:{:03d}, loss_gen: {:.6f}, tok_acc: {:.6f}, The min loss: {:.6f}".format(
                epoch, global_step + i + 1, int(e-s), metrics["train_loss_gen"],
                metrics["train_tok_acc"], metrics["min_train_loss"]))
            s = e
            metrics["train_loss_gen"] = 0.0
            metrics["train_tok_acc"] = 0.0
    metrics["global_step"] += (i + 1)

def val(model, epoch, data_loader, metrics, writer, print_step):
    s = time.time()
    model.eval()
    print_step = print_step
    i = 0
    with torch.no_grad():
        for i, (uinds, iinds, ratings, tips_vec) in enumerate(data_loader):
            expalin_scores, explain_hat = model(uinds, iinds, tips_vec)
            loss_per_tok, nll_loss, target_tokens, correct = model.loss_gen(expalin_scores, explain_hat, tips_vec)
            metrics["eval_loss_gen"] += loss_per_tok.item()
            metrics["eval_tok_acc"] += (correct / target_tokens)
            if (i+1) % print_step == 0:
                e = time.time()
                print("eval:[{:06d}] time_use:{:03d}, loss_gen: {:.6f}, tok_acc: {:.6f}, min_eval_loss: {:.6f}".format(
                    (i+1), int(e-s), metrics["eval_loss_gen"] / (i+1),
                    metrics["eval_tok_acc"] / (i+1), metrics["min_eval_loss"]))
                s = e
    metrics["eval_loss_gen"] /= (i+1)
    metrics["eval_tok_acc"] /= (i+1)
    writer.add_scalar("eval/eval_loss_gen", metrics["eval_loss_gen"], global_step=epoch)
    writer.add_scalar("eval/eval_tok_acc", metrics["eval_tok_acc"], global_step=epoch)

def main(opt):
    use_gnn = opt["use_gnn"]
    use_relation = opt["use_relation"]
    use_knowledge = opt["use_knowledge"]
    use_copy = opt["use_copy"]
    gnn = "_gnn" if use_gnn else ""
    relation = "_relation" if use_relation else ""
    knowledge = "_knowledge" if use_knowledge else ""
    copy = "_copy" if use_copy else ""
    the_time = time.strftime('_%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))

    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    model_dir = os.path.join(save_dir, data_name, "models_test_seq2seq{}{}{}{}".format(gnn, relation, knowledge, copy))
    log_dir = os.path.join(save_dir, data_name, "logs_test_seq2seq{}{}{}{}".format(gnn, relation, knowledge, copy), "{}".format(the_time))
    best_model_path = os.path.join(model_dir, "best_model.ckpt")
    device = opt["device"]
    epochs = opt["epochs"]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    prepare_datas(opt)
    toolkits = opt["toolkits"]
    # return
    data_manager = RecDatasetManager(opt)
    model = Seq2Seq(opt)
    model.to(device)

    if opt["run_type"] == "train":
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        init_epoch = 0
        metrics = {
            "train_tok_acc":0.0, 
            "train_loss_gen":0.0,
            "eval_tok_acc":0.0, 
            "eval_loss_gen":0.0,
            "min_train_loss":10e6,
            "min_eval_loss":10e6,
            "global_step":0
        }
    
        if os.path.exists(best_model_path) and not opt["new_train"]:
            print("loading best model from {}".format(best_model_path))
            ckpt = torch.load(best_model_path)
            model.load_state_dict(ckpt['model_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            init_epoch = ckpt["epoch"] + 1
            metrics["global_step"] = ckpt["global_step"] + 1
            metrics["min_train_loss"] = ckpt["min_train_loss"]
            metrics["min_eval_loss"] = ckpt["min_eval_loss"]

    # uinds = torch.Tensor(list(range(8)))
    # iinds = torch.Tensor(list(range(8)))
    # inputs = (uinds, iinds)
    # writer.add_graph(model, inputs)
        for epoch in range(init_epoch, epochs):
            train(model, optimizer, epoch, data_manager.train_loader, metrics, writer, opt["train_print_step"], toolkits)
            val(model, epoch, data_manager.val_loader, metrics, writer, opt["eval_print_step"])
            if metrics["eval_loss_gen"] < metrics["min_eval_loss"]:
                #save model
                metrics["min_eval_loss"] = metrics["eval_loss_gen"]
                save_dict = {"epoch":epoch, "global_step":metrics["global_step"], "model_dict":model.state_dict(), "min_train_loss":metrics["min_train_loss"],
                                "min_eval_loss":metrics["min_eval_loss"], "optimizer":optimizer.state_dict()}
                model_save_path = os.path.join(model_dir, "epoch_{}_loss_{:.6f}.ckpt".format(epoch, metrics["min_eval_loss"]))
                torch.save(save_dict, model_save_path)
                torch.save(save_dict, best_model_path)
                
                #只保留5个模型
                models_name = os.listdir(model_dir)
                if len(models_name) > 5:
                    models = [os.path.join(model_dir, name) for name in models_name]
                    models.sort(key=lambda x: os.path.getctime(x))
                    os.remove(models[0])
            
            metrics["eval_loss_rec"] = 0.0
            metrics["eval_loss_gen"] = 0.0
            metrics["eval_tok_acc"] = 0.0
        writer.close()
    else:
        with torch.no_grad():
            if os.path.exists(best_model_path):
                print("loading best model from {}".format(best_model_path))
                ckpt = torch.load(best_model_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(ckpt['model_dict'])
                model.eval()
                post = "{}{}{}{}".format(gnn, relation, knowledge, copy)
                system_sum_dir = os.path.join(save_dir, data_name, "tests_seq2seq{}".format(post), "system")
                model_sum_dir = os.path.join(save_dir, data_name, "tests_seq2seq{}".format(post), "model")
                mae_path = os.path.join(save_dir, data_name, "tests{}".format(post), "mae.txt")
                if not os.path.exists(system_sum_dir):
                    os.makedirs(system_sum_dir)
                if not os.path.exists(model_sum_dir):
                    os.makedirs(model_sum_dir)

                index = 0
                mae = 0.0
                rmse = 0.0
                num_ratings = 0
                for i, (uinds, iinds, ratings, tips_vec, tips) in tqdm(enumerate(data_manager.test_loader), total=len(data_manager.test_dataset) // opt["batch_size"]):
                    expalin_scores, explain_hat = model(uinds, iinds)
                    for tip_vec, pre_vec in zip(tips_vec, explain_hat):
                        ref_path = os.path.join(model_sum_dir, 'model.{}'.format(index))
                        gen_path = os.path.join(system_sum_dir, 'system.{}'.format(index))
                        with open(ref_path, 'w', encoding="utf8") as f_ref, open(gen_path, 'w', encoding="utf-8") as f_gen:
                            tip_vec = tip_vec.tolist()
                            pre_vec = pre_vec.tolist()
                            end_ind = opt["toolkits"].end_ind
                            try:
                                tip_vec = tip_vec[:tip_vec.index(end_ind)]
                            except Exception: pass
                            try:
                                pre_vec = pre_vec[:pre_vec.index(end_ind)]
                            except Exception: pass
                            f_ref.write(' '.join(map(str, tip_vec)))
                            f_gen.write(' '.join(map(str, pre_vec)))
                            index += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(
        run_type="train",
        new_train=True,
        data_source="Amazon",
        data_path="dataset/Musical_Instruments_5.json",
        splits="8:1:1",
        save_dir="saved",
        conceptnet_dir="saved/conceptnet",
        conceptnet_emb_type="float32",
        tokenizer="nltk",
        dict_language="english",
        min_word_freq=20,
        fp16=True,
        epochs=50,
        train_print_step=300,
        eval_print_step=10,
        batch_size=8,
        dropout=0.,
        num_heads=2,
        words_topk=10,
        user_topk=30,
        min_support = 0.,
        min_conf = 0.2,
        min_tip_len=5,
        rec_topk=10,
        w2v_emb_size=64,
        bilstm_hidden_size=32,
        hidden_size=64,
        bilstm_num_layers=2,
        gru_num_layers=2,
        max_text_len=256,
        max_sent_len=16,
        max_tip_len=64,
        max_copy_len=256,
        max_neighbors=30,
        use_gnn=False,
        use_relation=False,
        use_knowledge=False,
        use_copy=False,
    )
    opt = vars(parser.parse_args())
    opt["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.set_device(7)
    main(opt)