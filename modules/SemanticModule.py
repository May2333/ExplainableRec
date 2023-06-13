import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
import numpy as np
import os
import math

"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

class SemanticModule(nn.Module):
    def __init__(self, opt):
        super(SemanticModule, self).__init__()
        self.device = opt.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dict = opt["dict"]
        self.toolkits = opt["toolkits"]
        self.null_ind = self.toolkits.tok2ind(self.dict.null_tok)
        self.vocab_len = len(opt.get("dict"))
        self.w2v_emb_size = opt.get("w2v_emb_size", 300)
        self.wv2_weight_path = opt.get("w2v_weight_path", None)
        self.conceptnet_len = opt.get("conceptnet_len")
        self.conceptnet_emb_size = opt.get("conceptnet_emb_size", 300)
        self.conceptnet_emb_path = opt.get("conceptnet_emb_path")
        self.conceptnet_neibors_emb_path = opt.get("conceptnet_neibors_emb_path")
        self.hidden_size = opt.get("hidden_size")
        self.words_topk = opt.get("words_topk", 10)
        self.bilstm_hidden_size = opt.get("bilstm_hidden_size", 256)
        self.bilstm_num_layers = opt.get("bilstm_num_layers", 2)
        self.dropout = opt.get("dropout", 0)
        self.num_heads = opt.get("num_heads", 2)
        self.max_text_len = opt.get("max_text_len", 64)
        self.max_sent_len = opt.get("max_sent_len", 16)
        self.emb_norm1 = nn.LayerNorm(self.w2v_emb_size)
        self.emb_norm2 = nn.LayerNorm(self.w2v_emb_size)
        self.emb_norm3 = nn.LayerNorm(self.conceptnet_emb_size)
        self.out_norm1 = nn.LayerNorm(2 * self.bilstm_hidden_size)
        self.out_norm2 = nn.LayerNorm(2 * self.bilstm_hidden_size)
        self.out_norm3 = nn.LayerNorm(2 * self.bilstm_hidden_size)
        self.word2vec_encoder = Word2VecEncoder(self.w2v_emb_size, self.vocab_len, self.wv2_weight_path)
        self.conceptnet_encoder = ConceptNetEncoder(self.conceptnet_emb_size, self.hidden_size, self.dropout, self.conceptnet_len,
                                                        self.conceptnet_emb_path, self.conceptnet_neibors_emb_path, self.words_topk, self.device)
        self.multi_head_att = MultiHeadAttention(n_heads=self.num_heads, qdim=4 * self.bilstm_hidden_size, kdim=2 * self.bilstm_hidden_size,
                                                    vdim=2*self.bilstm_hidden_size , hdim=self.hidden_size, dropout=self.dropout)
        self.text_bilstm = LSTM_encoder(in_size=self.w2v_emb_size, hidden_size=self.bilstm_hidden_size,
                                    num_layers=self.bilstm_num_layers, dropout=self.dropout, device=self.device, biflag=True)
        self.sent_bilstm = LSTM_encoder(in_size=self.w2v_emb_size, hidden_size=self.bilstm_hidden_size,
                                    num_layers=self.bilstm_num_layers, dropout=self.dropout, device=self.device, biflag=True)
        self.conceptnet_bilstm = LSTM_encoder(in_size=self.conceptnet_emb_size, hidden_size=self.bilstm_hidden_size,
                                    num_layers=self.bilstm_num_layers, dropout=self.dropout, device=self.device, biflag=True)
    
    def sent_embedding(self, sent_vec):
        sent_emb = self.word2vec_encoder(sent_vec)
        return sent_emb.mean(dim=-2)
    
    def sem_out(self, hn, biflag=True):
        batch_size = hn.shape[1]
        num_layers = self.bilstm_num_layers
        num_directions = 2 if biflag else 1
        hn = hn.transpose(0, 1).reshape(batch_size, num_layers, -1)[:, -1, :]

        return hn

    def forward(self, text_vec, text_lens, sent_vec, sent_lens, conceptnet_text_vec):
        w2v_text_emb = self.word2vec_encoder(text_vec)
        w2v_sent_emb = self.sent_embedding(sent_vec)
        conceptnet_emb = self.conceptnet_encoder(conceptnet_text_vec)

        # text_packed = rnn.pack_padded_sequence(w2v_text_emb, text_lens, batch_first=True)
        # sent_packed = rnn.pack_padded_sequence(w2v_sent_emb, sent_lens, batch_first=True)
        # conceptnet_packed = rnn.pack_padded_sequence(conceptnet_emb, text_lens, batch_first=True)

        text_out, text_hn = self.text_bilstm(w2v_text_emb)
        sent_out, sent_hn = self.sent_bilstm(w2v_sent_emb)
        conceptnet_out, _ = self.conceptnet_bilstm(conceptnet_emb)

        sem_g = self.sem_out(text_hn)           #batch_size * 2h_size
        # return sem_g
        sem_s = self.sem_out(sent_hn)           #batch_size * 2h_size
        # sem_ca = self.sem_out(conceptnet_out, text_lens)    #batch_size * 2h_size 
        sem_c = torch.cat([sem_g, sem_s], dim=-1)           #batch_size * 4h_size
        mask = (text_vec != self.null_ind)
        # sem_out = self.multi_head_att(sem_c.unsqueeze(dim=1), text_out, text_out, mask)
        sem_out = self.multi_head_att(sem_c.unsqueeze(dim=1), text_out, conceptnet_out, mask)
        return sem_out

class Word2VecEncoder(nn.Module):
    def __init__(self, emb_size, vocab_len, wv2_weight_path):
        super(Word2VecEncoder, self).__init__()
        if os.path.exists(wv2_weight_path):
            print("Load word2vec weight from exist file {}".format(wv2_weight_path))
            weight = torch.from_numpy(np.load(wv2_weight_path)).float()
            self.w2v_emb = nn.Embedding.from_pretrained(weight, freeze=False)
            # weight = torch.from_numpy(np.load(wv2_weight_path)).float()
            # self.w2v_emb = nn.Embedding(vocab_len, emb_size)
            # self.w2v_emb.from_pretrained(weight)
        else:
            print("Can not find pretrained word2vec weight file, and it will init randomly")
            self.w2v_emb = nn.Embedding(vocab_len, emb_size)
            nn.init.xavier_normal_(self.w2v_emb.weight)
    def forward(self, text_vec):
        emb = self.w2v_emb(text_vec)
        return emb

class ConceptNetEncoder(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout, conceptnet_len, conceptnet_emb_path, conceptnet_neibors_emb_path, topk, device):
        super(ConceptNetEncoder, self).__init__()
        self.emb_size = emb_size
        self.conceptnet_len = conceptnet_len
        self.topk = topk
        if os.path.exists(conceptnet_emb_path):
            print("Load conceptnet numberbatch weight from exist file {}".format(conceptnet_emb_path))
            weight = torch.from_numpy(np.load(conceptnet_emb_path))
            self.conceptnet_emb = nn.Embedding.from_pretrained(weight)
            # weight = torch.from_numpy(np.load(conceptnet_emb_path))
            # self.conceptnet_emb = nn.Embedding(conceptnet_len, emb_size)
            # self.conceptnet_emb.from_pretrained(weight)
        else:
            print("Can not find pretrained conceptnet numberbatch weight file, and it will init randomly")
            self.conceptnet_emb = nn.Embedding(self.conceptnet_len, self.emb_size)
            nn.init.xavier_normal_(self.conceptnet_emb.weight)

        if os.path.exists(conceptnet_neibors_emb_path):
            print("Load conceptnet neighbor weight from exist file {}".format(conceptnet_neibors_emb_path))
            self.neighbors = torch.from_numpy(np.load(conceptnet_neibors_emb_path)).to(device)
        else:
            raise("no neighbors")
        
        self.self_att = SelfAttentionLayer(self.emb_size, self.emb_size, dropout=dropout)

    def forward(self, conceptnet_text_vec):
        bs = conceptnet_text_vec.size(0)
        neighbors = self.neighbors[conceptnet_text_vec]
        # emb = self.conceptnet_emb(conceptnet_text_vec)
        # matchs = []
        # for b in range(bs):
        #     match = emb[b].unsqueeze(0).matmul(self.conceptnet_emb.weight.t())
        #     matchs.append(torch.topk(match, k=self.topk, dim=-1).indices)
        #     del match
        # matchs = torch.cat(matchs, dim=0)
        # match = F.softmax(torch.matmul(emb, self.conceptnet_emb.weight.t()), dim=-1)
        # match = torch.topk(match, k=self.topk, dim=-1).indices
        att_emb = self.self_att(self.conceptnet_emb(neighbors))
        return att_emb

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_normal_(self.a.data)
        nn.init.xavier_normal_(self.b.data)

    def forward(self, h):
        N = h.shape[-2]
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).transpose(-1, -2)
        attention = F.softmax(e, dim=-1)
        return torch.matmul(attention, h).squeeze(-2)

class LSTM_encoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, dropout, device, biflag=True):
        super(LSTM_encoder, self).__init__()
        self.device = device
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.biflag = biflag
        self.num_directions = 2 if self.biflag else 1
        self.lstm = nn.LSTM(input_size=self.in_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, dropout=self.dropout, bidirectional=self.biflag)
        
    def init_hidden_state(self, batch_size):
        return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device))

    def forward(self, input_data):
        if isinstance(input_data, torch.Tensor):
            batch_size = input_data.shape[0]
        else:
            batch_size = input_data.batch_sizes[0].item()
        h_c = self.init_hidden_state(batch_size)
        out, (hn, cn) = self.lstm(input_data, h_c)
        return out, hn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, qdim, kdim, vdim, hdim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = hdim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(qdim, hdim)
        self.k_lin = nn.Linear(kdim, hdim)
        self.v_lin = nn.Linear(vdim, hdim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(hdim, hdim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key, value, mask=None):
        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, qdim = query.size()
        # assert qdim == 4 * self.dim, \
        #     f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        def neginf(dtype):
            """Returns a representable finite number near -inf for a dtype."""
            if dtype is torch.float16:
                return -NEAR_INF_FP16
            else:
                return -NEAR_INF

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, kdim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2).contiguous()
            .view(batch_size, query_len, self.dim)
        )

        out = self.out_lin(attentioned)

        return out.squeeze(1)
