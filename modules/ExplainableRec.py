import torch
from torch import nn
from modules.SemanticModule import SemanticModule
from modules.GATModule import GATModule
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

class ExplainableRec(nn.Module):
    def __init__(self, opt):
        super(ExplainableRec, self).__init__()
        self.use_gnn = opt["use_gnn"]
        self.use_relation = opt["use_relation"]
        self.use_knowledge = opt["use_knowledge"]
        self.use_copy = opt["use_copy"]
        self.dict = opt["dict"]
        self.emb_size = opt["w2v_emb_size"]
        self.toolkits = opt["toolkits"]
        self.hidden_size = opt["hidden_size"]
        self.dropout = opt["dropout"]
        self.max_tip_len = opt["max_tip_len"]
        self.max_copy_len = opt["max_copy_len"]
        self.gru_num_layers = opt["gru_num_layers"]
        self.device = opt["device"]
        self.end_ind = self.toolkits.tok2ind(self.dict.end_tok)
        self.null_ind = self.toolkits.tok2ind(self.dict.null_tok)
        if self.use_knowledge:
            self.semantic_encoder = SemanticModule(opt)
            self.embedding_layer = self.semantic_encoder.word2vec_encoder.w2v_emb
        else:
            self.semantic_encoder = None
            self.embedding_layer = nn.Embedding(len(self.dict), self.emb_size)
            nn.init.xavier_normal_(self.embedding_layer.weight)

        self.gat = GATModule(opt, self.semantic_encoder)
        self.recommender = Recommender(self.hidden_size, self.dropout)
        self.gru_explainer = GruExplainer(
            embedding_layer=self.embedding_layer,
            word_dict=self.dict,
            toolkits=self.toolkits,
            max_tip_len=self.max_tip_len,
            max_copy_len=self.max_copy_len,
            hidden_size=self.hidden_size,
            num_layers=self.gru_num_layers,
            device=self.device,
            dropout=0.,
            use_copy=self.use_copy
        )
        self.h0_layer = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.h0_norm = nn.LayerNorm(self.hidden_size)
        nn.init.xavier_normal_(self.h0_layer.weight)

        self.rec_criterion = nn.MSELoss()
        self.gen_criterion = nn.CrossEntropyLoss(ignore_index=self.null_ind, reduction='sum')
    
    def get_copy_src_mask(self, users_ind, items_ind):
        bs = users_ind.shape[0]
        masks = torch.zeros([bs, len(self.dict)]).to(self.device)
        for b, (uind, iind) in enumerate(zip(users_ind, items_ind)):
            src_vec = self.gat.get_user_vectors(uind)[0] + self.gat.get_item_vectors(iind)[0]
            masks[b][src_vec] = 1
        masks[:, 0] = 0
        return masks

    def forward(self, users_ind, items_ind, ys=None):
        users_pref, items_pref, relations_pref = self.gat(users_ind, items_ind)
        h0 = self.h0_layer(torch.cat((users_pref, items_pref, relations_pref), dim=1))
        h0 = torch.tanh(h0)
        r_hat = self.recommender(users_pref, items_pref, relations_pref)
        masks = None, None
        if self.use_copy:
            masks = self.get_copy_src_mask(users_ind, items_ind)
        scores, predicts = self.gru_explainer(masks, h0, ys)
        return r_hat, scores, predicts
    
    def loss_rec(self, r_hat, r_true):
        return self.rec_criterion(r_hat, r_true)

    def loss_gen(self, scores, y_hat, y_true):
        nll_loss = self.gen_criterion(scores.view(-1, scores.size(-1)), y_true.view(-1))
        notnull = y_true.ne(self.toolkits.tok2ind(self.dict.null_tok))
        target_tokens = notnull.long().sum().item()
        correct = ((y_true == y_hat) * notnull).sum().item()
        # average loss per token
        loss_per_tok = nll_loss / target_tokens
        return loss_per_tok, nll_loss, target_tokens, correct


class Recommender(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(Recommender, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.linear2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(2 * self.hidden_size, 1)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)

    
    def forward(self, users_pref, items_pref, relations_pref):
        i_uv = self.linear1(torch.cat((users_pref, items_pref), dim=1))
        i_uv = _normalize(self.dropout(i_uv), self.norm1)
        i_uv = F.relu(i_uv)

        z_h = i_uv * relations_pref

        z_m = self.linear2(torch.cat((i_uv, relations_pref), dim=1))
        z_m = _normalize(self.dropout(z_m), self.norm2)
        z_m = F.relu(z_m)

        r_uv = self.linear3(torch.cat((z_h, z_m), dim=1))

        return r_uv.squeeze()   

class GruExplainer(nn.Module):
    def __init__(self, embedding_layer, word_dict, toolkits, max_tip_len, max_copy_len, hidden_size, num_layers, device, dropout=0., use_copy=True):
        super(GruExplainer, self).__init__()
        self.use_copy = use_copy
        self.dict = word_dict
        self.toolkits = toolkits
        self.start_ind = toolkits.tok2ind(word_dict.start_tok)
        self.end_ind = toolkits.tok2ind(word_dict.end_tok)
        self.null_ind = toolkits.tok2ind(word_dict.null_tok)
        self.max_tip_len = max_tip_len
        self.max_copy_len = max_copy_len
        self.device = device
        self.hidden_size = hidden_size
        self.emb_size = embedding_layer.embedding_dim
        self.embedding_layer = embedding_layer
        self.dropout_layer = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(self.emb_size)
        self.gru_decoder = nn.GRU(self.emb_size, hidden_size, batch_first=True, dropout=dropout)
        self.out_linear = nn.Linear(hidden_size, self.emb_size)
        self.norm2 = nn.LayerNorm(self.emb_size)
        self.copy_trans = nn.Linear(2 * hidden_size, self.emb_size)
        self.norm3 = nn.LayerNorm(self.emb_size)

    def neginf(self, dtype):
            """Returns a representable finite number near -inf for a dtype."""
            if dtype is torch.float16:
                return -NEAR_INF_FP16
            else:
                return -NEAR_INF

    def decode_forced(self, masks, ys, h0):
        bs = ys.shape[0]
        starts = torch.LongTensor([self.start_ind]).repeat((bs, 1)).to(self.device)
        seqlen = ys.shape[1]
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat((starts, inputs), 1)
        x = self.embedding_layer(inputs)
        # x = _normalize(x, self.norm1)
        hs, hn = self.gru_decoder(x, h0.unsqueeze(0))
        x1_scores = self.out_linear(hs)
        # x1_scores = _normalize(x1_scores, self.norm1)
        #*****************
        x1_scores = F.relu(F.linear(x1_scores, self.embedding_layer.weight))
        if not self.use_copy:
            _, predicts = x1_scores.max(dim=-1)
            return x1_scores, predicts
        masks = (masks == 0).unsqueeze(1).expand(-1, seqlen, -1)
        h0 = h0.view(bs, 1, -1).repeat(1, seqlen, 1)
        x2_scores = self.copy_trans(torch.cat((hs, h0), dim=-1))
        x2_scores = F.relu(F.linear(x2_scores, self.embedding_layer.weight))
        x2_scores = x2_scores.masked_fill_(masks, 0.0)
        
        scores = x1_scores + x2_scores
        _, predicts = scores.max(dim=-1)
        return scores, predicts
        #*****************

    def decode_greedy(self, masks, h0):
        bs = h0.shape[0]
        x = torch.LongTensor([self.start_ind]).repeat((bs, 1)).to(self.device)
        if masks is not None:
            masks = (masks == 0)
        scores = []
        hn = h0.unsqueeze(0)
        for i in range(self.max_tip_len):
            h = self.embedding_layer(x)
            # h = _normalize(h, self.norm1)
            # print(hn)
            hs, hn = self.gru_decoder(h, hn)
            hs = hs[:, -1, :]
            x1_scores =  self.out_linear(hs)
            x1_scores = F.linear(x1_scores, self.embedding_layer.weight)
            if self.use_copy:
                x2_scores = self.copy_trans(torch.cat((hs, h0), dim=-1))
                x2_scores = F.relu(F.linear(x2_scores, self.embedding_layer.weight))
                x2_scores = x2_scores.masked_fill_(masks, 0.0)
                score = x1_scores + x2_scores
            else:
                score = x1_scores
            _, predicts = score.max(dim=-1)
            scores.append(score.unsqueeze(1))
            x = torch.cat((x, predicts.unsqueeze(1)), dim=1)
            all_finished = ((x == self.end_ind).sum(dim=1) > 0).sum().item() == bs
            if all_finished:
                break

        return torch.cat(scores, dim=1), x[:, 1:]

    def forward(self, masks, h0, ys=None):
        if ys is not None:
            scores, predicts = self.decode_forced(masks, ys, h0)
        else:
            scores, predicts = self.decode_greedy(masks, h0)

        return scores, predicts