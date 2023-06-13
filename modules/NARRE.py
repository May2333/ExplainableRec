
import torch
import torch.nn.functional as F
import numpy as np

class ReviewEncoder(torch.nn.Module):
    def __init__(self, opt, preference_id_count: int, quality_id_count: int):
        super().__init__()

        self.preference_id_embedding = torch.nn.Embedding(preference_id_count, opt["hidden_size"])
        self.quality_id_embedding = torch.nn.Embedding(quality_id_count, opt["hidden_size"])
        self.opt = opt
        self.conv = torch.nn.Conv1d(
            in_channels=opt["w2v_emb_size"],
            out_channels=100,
            kernel_size=3,
            stride=1)
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=opt["max_tip_len"] - 3 + 1,
            stride=1)

        self.att_review = torch.nn.Linear(100, opt["hidden_size"])
        self.att_id = torch.nn.Linear(opt["hidden_size"], opt["hidden_size"], bias=False)
        self.att_layer = torch.nn.Linear(opt["hidden_size"], 1)

        self.top_linear = torch.nn.Linear(100, opt["hidden_size"])
        self.dropout = torch.nn.Dropout(opt["dropout"])

    def forward(self, review_emb, preference_id, quality_id):
        """
        Input Size:
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, Review Count)
            (Batch Size, Review Count)

        Output Size:
            (Batch Size, Id Dim)
        """

        preference_id_emb = self.preference_id_embedding(preference_id).view(-1, self.opt["hidden_size"])
        quality_id_emb = self.quality_id_embedding(quality_id)

        batch_size = review_emb.shape[0]
        review_in_one = review_emb.view(-1, self.opt["max_tip_len"], self.opt["w2v_emb_size"])
        review_in_one = review_in_one.permute(0, 2, 1)
        review_conv = F.relu(self.conv(review_in_one))
        review_conv = self.max_pool(review_conv).view(-1, 100)
        review_in_many = review_conv.view(batch_size, 25, -1)

        review_att = self.att_review(review_in_many)
        id_att = self.att_id(quality_id_emb)
        att_weight = self.att_layer(F.relu(review_att + id_att))
        att_weight = F.softmax(att_weight, dim=1)
        att_out = (att_weight * review_in_many).sum(1)

        feature = self.dropout(att_out)
        feature = self.top_linear(feature)
        feature = preference_id_emb + feature

        tip_index = att_weight.argmax(1).unsqueeze(2)
        return feature, tip_index.expand(batch_size, 1, self.opt["max_tip_len"])


class LatentFactor(torch.nn.Module):
    def __init__(self, opt, user_nums, item_nums):
        super().__init__()
        self.linear = torch.nn.Linear(opt["hidden_size"], 1)
        self.b_user = torch.nn.Parameter(torch.randn([user_nums]), requires_grad=True)
        self.b_item = torch.nn.Parameter(torch.randn([item_nums]), requires_grad=True)

    def forward(self, user_feature, user_id, item_feature, item_id):
        """
        Input Size:
            (Batch Size, Id Dim)
            (Batch Size, 1)
            (Batch Size, Id Dim)
            (Batch Size, 1)

        Output Size:
            (Batch Size, 1)
        """
        dot = user_feature * item_feature
        predict = self.linear(dot) + self.b_user[user_id.view(-1, 1)] + self.b_item[item_id.view(-1, 1)]
        return predict


class NarreModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.toolkits = opt["toolkits"]
        self.user_nums = len(self.toolkits.user2ind_dict)
        self.item_nums = len(self.toolkits.item2ind_dict)
        weight = torch.from_numpy(np.load(opt["w2v_weight_path"])).float()
        self.word_embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)

        self.user_review_layer = ReviewEncoder(opt, self.user_nums+1, self.item_nums+1)
        self.item_review_layer = ReviewEncoder(opt, self.item_nums+1, self.user_nums+1)

        self.predict_linear = LatentFactor(opt, self.user_nums, self.item_nums)

    def forward(self, user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review):
        """
        Input Size:
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, 1)
            (Batch Size, Review Count)
            (Batch Size, Review Count, Review Length, Word Dim)
            (Batch Size, 1)
            (Batch Size, Review Count)

        Output Size:
            (Batch Size, 1)
        """

        user_review_emb = self.word_embedding(user_review)
        user_feature, user_tip_index = self.user_review_layer(user_review_emb, user_id, item_id_per_review)

        item_review_emb = self.word_embedding(item_review)
        item_feature, item_tip_index = self.item_review_layer(item_review_emb, item_id, user_id_per_review)

        predict = self.predict_linear(user_feature, user_id, item_feature, item_id)
        return predict.squeeze(), user_review.gather(1, user_tip_index).squeeze(), item_review.gather(1, item_tip_index).squeeze()
