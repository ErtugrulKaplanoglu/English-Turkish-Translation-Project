
import json

#%%
'''
Word level tokeniztation
'''

def normalize_text(text):
    text = text.lower().strip()

    result = ''
    prev_space = False

    for ch in text:
        if ch == ' ':
            if not prev_space:
                result += ch
            prev_space = True
        else:
            result += ch
            prev_space = False

    return result


def word_tokenizer(text):
    
    text = normalize_text(text)

    tokens = []
    current = ''
    punctuation = {'.', ',', '!', '?'}

    for ch in text:
        if ch == ' ':
            if current:
                tokens.append(current)
                current = ''
        elif ch in punctuation:
            if current:
                tokens.append(current)
            tokens.append(ch)
            current = ''
        else:
            current += ch

    if current:
        tokens.append(current)

    return tokens


#%%
'''
BPE strucute tokenizer
'''

SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']
WORD_END = '</w>'


def _word_to_symbols(word):
    
    return list(word) + [WORD_END]


def _get_pair_counts(vocab_symbols):
    
    pair_counts = {}

    for symbols, freq in vocab_symbols.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq

    return pair_counts


def _merge_once(pair, vocab_symbols):
    
    a, b = pair
    merged = a + b

    new_vocab = {}
    for symbols, freq in vocab_symbols.items():
        symbols = list(symbols)
        i = 0
        new_seq = []

        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(symbols[i])
                i += 1

        new_seq = tuple(new_seq)
        new_vocab[new_seq] = new_vocab.get(new_seq, 0) + freq

    return new_vocab


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        self.is_trained = False

    def train_from_texts(self, texts, min_frequency=2, max_merges=10000):
        # 1) word frekansı
        word_freq = {}
        for t in texts:
            for tok in word_tokenizer(t):
                word_freq[tok] = word_freq.get(tok, 0) + 1

        # 2) başlangıç vocab: word -> char symbols
        vocab_symbols = {}
        for w, f in word_freq.items():
            if f < min_frequency:
                continue

            # noktalama için BPE uygulamayalım: tek token kalsın
            if len(w) == 1 and w in {'.', ',', '!', '?'}:
                sym = (w, WORD_END)
            else:
                sym = tuple(_word_to_symbols(w))

            vocab_symbols[sym] = vocab_symbols.get(sym, 0) + f

        # 3) merge döngüsü
        self.merges = []
        for _ in range(max_merges):
            pair_counts = _get_pair_counts(vocab_symbols)
            if not pair_counts:
                break

            # en sık pair
            best_pair = None
            best_count = 0
            for pair, cnt in pair_counts.items():
                if cnt > best_count:
                    best_pair = pair
                    best_count = cnt

            if best_pair is None or best_count < min_frequency:
                break

            vocab_symbols = _merge_once(best_pair, vocab_symbols)
            self.merges.append(best_pair)

        # 4) token set'i çıkar
        tokens = set()
        for sym_seq in vocab_symbols.keys():
            for s in sym_seq:
                tokens.add(s)

        for st in SPECIAL_TOKENS:
            tokens.add(st)

        # 5) id map
        tokens = sorted(list(tokens))
        self.token_to_id = {}
        for i, t in enumerate(tokens):
            self.token_to_id[t] = i

        self.id_to_token = {}
        for t, i in self.token_to_id.items():
            self.id_to_token[int(i)] = t

        self.is_trained = True

    def save(self, path):
        data = {
            'merges': self.merges,
            'token_to_id': self.token_to_id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.merges = [tuple(x) for x in data.get('merges', [])]
        self.token_to_id = data.get('token_to_id', {})

        self.id_to_token = {}
        for t, i in self.token_to_id.items():
            self.id_to_token[int(i)] = t

        self.is_trained = True

    def _apply_merges(self, symbols):
        for a, b in self.merges:
            merged = a + b
            i = 0
            new_seq = []

            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(symbols[i])
                    i += 1

            symbols = new_seq

        return symbols

    def tokenize_word(self, word):
        symbols = _word_to_symbols(word)
        symbols = self._apply_merges(symbols)
        return symbols

    def tokenize(self, text):
        tokens = []
        for w in word_tokenizer(text):
            if len(w) == 1 and w in {'.', ',', '!', '?'}:
                tokens.append(w)
                continue

            pieces = self.tokenize_word(w)
            for p in pieces:
                if p != WORD_END:
                    tokens.append(p)

        return tokens

    def encode_ids(self, text):
        if not self.is_trained:
            raise ValueError('BPE tokenizer trained/loaded değil.')

        toks = self.tokenize(text)
        unk_id = self.token_to_id.get('<unk>', 0)
        ids = []
        for t in toks:
            ids.append(self.token_to_id.get(t, unk_id))
        return ids


bpe_en = BPETokenizer()
bpe_tr = BPETokenizer()



def _read_lines(file_paths):
    texts = []
    for fp in file_paths:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def train_bpe_en_from_files(file_paths, min_frequency=2, max_merges=10000):
    texts = _read_lines(file_paths)
    bpe_en.train_from_texts(texts=texts, min_frequency=min_frequency, max_merges=max_merges)


def train_bpe_tr_from_files(file_paths, min_frequency=2, max_merges=10000):
    texts = _read_lines(file_paths)
    bpe_tr.train_from_texts(texts=texts, min_frequency=min_frequency, max_merges=max_merges)


def save_bpe_en(path):
    bpe_en.save(path)

def load_bpe_en(path):
    bpe_en.load(path)

def save_bpe_tr(path):
    bpe_tr.save(path)

def load_bpe_tr(path):
    bpe_tr.load(path)


def tokenize_en(text):
    return bpe_en.tokenize(text)

def tokenize_tr(text):
    return bpe_tr.tokenize(text)

def encode_ids_en(text):
    return bpe_en.encode_ids(text)

def encode_ids_tr(text):
    return bpe_tr.encode_ids(text)
