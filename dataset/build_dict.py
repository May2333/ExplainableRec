
import json
import os
from tqdm import tqdm
from collections import defaultdict
import string
import re
import pickle
#copy from parLAI
def escape(s):
    """
    Replace potential special characters with escaped version.

    For example, \n => \\n and \t => \\t

    :param s:
        string to escape
    """
    return s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')


def unescape(s):
    """
    Revert escaped characters back to their special version.

    For example, \\n => \n and \\t => \t

    :param s:
        string to unescape
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

class Dict(object):
    def __init__(self, opt):
        self.opt = opt
        self.minfreq = opt.get("min_word_freq", 0)
        self.maxtokens = opt.get("max_tokens", -1)
        self.null_tok = opt.get("null_tok", "__null__")
        self.unk_tok = opt.get("unk_tok", "__unk__")
        self.start_tok = opt.get("start_tok", "__start__")
        self.end_tok = opt.get("end_tok", "__end__")
        self.tok2ind = {}
        self.ind2tok = {}
        self.freq = defaultdict(int)
        self.tokenizer = opt['tokenizer']
        self.min_tip_len = opt["min_tip_len"]
        self.stop_words = self.load_stop_words()
        try:
            self.tokenizer_fun = getattr(self, self.tokenizer + '_tokenize')
        except AttributeError:
            raise AttributeError(
                'tokenizer type {} not yet supported'.format(self.tokenizer))

        if self.tokenizer == 'nltk':
            try:
                import nltk
            except ImportError:
                raise ImportError('Please install nltk (pip install nltk)')
            # nltk-specific setup
            st_path = 'tokenizers/punkt/{0}.pickle'.format(opt['dict_language'])
            try:
                self.sent_tok = nltk.data.load(st_path)
            except LookupError:
                nltk.download('punkt')
                self.sent_tok = nltk.data.load(st_path)
            self.word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()
    
    def __len__(self):
        return len(self.tok2ind)

    def load_stop_words(self):
        stop_words = {}
        with open("dataset/stopwords_en.txt") as f:
            for line in f:
                line = line.strip("\n")
                stop_words[line] = 1
        return stop_words

    def clean_sentence(self, sent):
        translator = str.maketrans('', '', string.punctuation)
        sent = sent.translate(translator)
        res = []
        for word in sent.split():
            if word in self.stop_words:
                continue
            res.append(word)
        return " ".join(res)
    
    def clean_review(self, review):
        res = []
        sents = self.sent_tok.tokenize(review)
        for sent in sents:
            res.append(self.clean_sentence(sent))
        return res

    def nltk_tokenize(self, text):
        return [token for sent in self.sent_tok.tokenize(text)
                for token in self.word_tok.tokenize(sent)]

    def tokenize(self, text, save_stop_words=False, lower=True):
        if lower:
            text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        res = []
        for word in text.split():
            if not save_stop_words:
                if word in self.stop_words:
                    continue
            res.append(word)
        return res
    
    #copy from parLAI
    def add_to_dict(self, tokens):
        """Build dictionary from the list of provided tokens."""
        for token in tokens:
            self.add_token(token)
            self.freq[token] += 1

    def add_token(self, word):
        if word not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[word] = index
            self.ind2tok[index] = word

    #copy from parLAI
    def sort(self, trim=True):
        """
        Sorts the dictionary, so that the elements with the lowest index have
        the highest counts. This reindexes the dictionary according to the
        sorted frequencies, breaking ties alphabetically by token.

        :param bool trim: If True, truncate the dictionary based on minfreq and
            maxtokens.
        """
        # sort first by count, then alphabetically
        if trim:
            self.remove_tail(self.minfreq)
        sorted_pairs = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        new_tok2ind = {}
        new_ind2tok = {}
        for i, (tok, _) in enumerate(sorted_pairs):
            new_tok2ind[tok] = i
            new_ind2tok[i] = tok
        self.tok2ind = new_tok2ind
        self.ind2tok = new_ind2tok
        if trim:
            self.resize_to_max(self.maxtokens)
        assert len(self.freq) == len(self.ind2tok) == len(self.tok2ind)
        return sorted_pairs

    def remove_tail(self, min_freq):
        """Remove elements below the frequency cutoff from the dictionary."""
        to_remove = []
        for token, freq in self.freq.items():
            if freq < min_freq:
                # queue up removals since can't mutate dict during iteration
                to_remove.append(token)

        for token in to_remove:
            del self.freq[token]
            idx = self.tok2ind.pop(token)
            del self.ind2tok[idx]

    def resize_to_max(self, maxtokens):
        """Trims the dictionary to the maximum number of tokens."""
        if maxtokens >= 0 and len(self.tok2ind) > maxtokens:
            for k in range(maxtokens, len(self.ind2tok)):
                v = self.ind2tok[k]
                del self.ind2tok[k]
                del self.tok2ind[v]
                del self.freq[v]

    def build(self, sort=True):
        save_dir = self.opt["save_dir"]
        data_path = self.opt["data_path"]
        data_name = os.path.basename(data_path).split('.')
        data_name = ".".join(data_name[:-1])
        save_dir = os.path.join(save_dir, data_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "dict")

        total = 0
        with open(data_path) as f:
            for line in f:
                total += 1
            self.opt["total_instances"] = total

        if os.path.exists(save_path):
            print("Dict {} is already exists".format(save_path))
            with open(save_path, 'r', encoding="utf-8") as f_dict:
                for ind, line in enumerate(f_dict.readlines()):
                    tok, cnt = line.split("\t")
                    tok = unescape(tok)
                    self.ind2tok[ind] = tok
                    self.tok2ind[tok] = ind
                    self.freq[tok] = int(cnt.strip('\n'))

        else:
            if self.null_tok:
                self.add_token(self.null_tok)
                self.freq[self.null_tok] = 1000000003
            if self.unk_tok:
                self.add_token(self.unk_tok)
                self.freq[self.unk_tok] = 1000000002
            if self.start_tok:
                self.add_token(self.start_tok)
                self.freq[self.start_tok] = 1000000001
            if self.end_tok:
                self.add_token(self.end_tok)
                self.freq[self.end_tok] = 1000000000

            print("Building dict from {}".format(data_path))
            if self.opt["data_source"] == "Amazon":
                text_field = "reviewText"
                tip_field = "summary"
            elif self.opt["data_source"] == "Yelp":
                text_field = ""
                tip_field = ""
                #TODO:  add yelp
            with open(data_path, 'r', encoding="utf-8") as f_data:
                for i, line in enumerate(tqdm(f_data, total=total)):
                    instance = json.loads(line)
                    text = instance[text_field]
                    tip = instance[tip_field]
                    if len(tip.split()) < self.min_tip_len:
                        for tip in self.sent_tok.tokenize(text):
                            if len(tip.split()) >= self.min_tip_len:
                                break
                    if len(tip.split()) < self.min_tip_len:
                        continue

                    text = text.strip()
                    tip = tip.strip()
                    self.add_to_dict(self.tokenize(text))
                    self.add_to_dict(self.tokenize(tip, save_stop_words=True))

            if sort:
                self.sort(trim=True)
            #save
            with open(save_path, 'w') as f_dict:
                for ind, tok in self.ind2tok.items():
                    cnt = self.freq[tok]
                    f_dict.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))
            
            # with open(os.path.join(save_dir, "dict.pikle"), 'wb') as f:
            #     pickle.dump(self, f)
