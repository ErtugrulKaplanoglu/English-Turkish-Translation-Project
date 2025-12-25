import os
import numpy as np
import tensorflow as tf
import sentencepiece as spm

class DotAttention(tf.keras.layers.Layer):
    def call(self, inputs):
        query, value = inputs
        scores = tf.matmul(query, value, transpose_b=True)   # (B, Tt, Ts)
        weights = tf.nn.softmax(scores, axis=-1)             # softmax over Ts
        context = tf.matmul(weights, value)                  # (B, Tt, H)
        return context

class Seq2SeqTranslator:
    def __init__(
        self,
        models_dir: str = "models",
        spm_file: str = "spm_bpe_shared_en_tr.model",
        en2tr_file_keras: str = "en2tr_lstm_attn_bpe.keras",
        tr2en_file_keras: str = "tr2en_lstm_attn_bpe.keras",
        # fallback weights-only (opsiyonel)
        en2tr_file_h5: str = "en2tr_lstm_attn_bpe.h5",
        tr2en_file_h5: str = "tr2en_lstm_attn_bpe.h5",
        # training ile aynı olmalı
        vocab_size: int = 16000,
        max_len: int = 30,
        emb_dim: int = 256,
        latent_dim: int = 256,
    ):
        self.models_dir = models_dir
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim

        self.spm_path = os.path.join(models_dir, spm_file)

        self.en2tr_keras_path = os.path.join(models_dir, en2tr_file_keras)
        self.tr2en_keras_path = os.path.join(models_dir, tr2en_file_keras)

        self.en2tr_h5_path = os.path.join(models_dir, en2tr_file_h5)
        self.tr2en_h5_path = os.path.join(models_dir, tr2en_file_h5)


        if not os.path.exists(self.spm_path):
            raise FileNotFoundError(f"Tokenizer bulunamadı: {self.spm_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.spm_path)

        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        # Load models
        self.en2tr = self._load_model(direction="en2tr")
        self.tr2en = self._load_model(direction="tr2en")

    # ---------- Model builder (for weights-only fallback) ----------
    def _build_seq2seq(self) -> tf.keras.Model:
        MAX_LEN = self.max_len
        VOCAB = self.vocab_size

        enc_in = tf.keras.layers.Input(shape=(MAX_LEN,), name="encoder_inputs")
        dec_in = tf.keras.layers.Input(shape=(MAX_LEN,), name="decoder_inputs")

        enc_emb = tf.keras.layers.Embedding(VOCAB, self.emb_dim, mask_zero=True, name="enc_emb")(enc_in)
        enc_lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True, name="encoder_lstm")
        enc_out, h, c = enc_lstm(enc_emb)

        dec_emb = tf.keras.layers.Embedding(VOCAB, self.emb_dim, mask_zero=True, name="dec_emb")(dec_in)
        dec_lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
        dec_out, _, _ = dec_lstm(dec_emb, initial_state=[h, c])

        ctx = DotAttention(name="dot_attention")([dec_out, enc_out])
        concat = tf.keras.layers.Concatenate(name="concat")([dec_out, ctx])
        out = tf.keras.layers.Dense(VOCAB, activation="softmax", name="softmax")(concat)

        return tf.keras.Model([enc_in, dec_in], out)

    def _load_model(self, direction: str) -> tf.keras.Model:
        # Prefer .keras
        if direction == "en2tr" and os.path.exists(self.en2tr_keras_path):
            return tf.keras.models.load_model(
                self.en2tr_keras_path,
                custom_objects={"DotAttention": DotAttention},
                compile=False
            )
        if direction == "tr2en" and os.path.exists(self.tr2en_keras_path):
            return tf.keras.models.load_model(
                self.tr2en_keras_path,
                custom_objects={"DotAttention": DotAttention},
                compile=False
            )

        weights_path = self.en2tr_h5_path if direction == "en2tr" else self.tr2en_h5_path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"{direction} için ne .keras ne de .h5 bulundu.\n"
                f"Beklenen: {self.en2tr_keras_path} / {self.en2tr_h5_path} (en2tr)\n"
                f"Beklenen: {self.tr2en_keras_path} / {self.tr2en_h5_path} (tr2en)"
            )

        model = self._build_seq2seq()
        model.load_weights(weights_path)
        return model

    def _encode_src(self, text: str) -> np.ndarray:
        ids = self.sp.encode(text, out_type=int)[: self.max_len]
        src = np.full((1, self.max_len), self.pad_id, dtype=np.int32)
        src[0, : len(ids)] = ids
        return src

    def _greedy_decode(self, model: tf.keras.Model, src: np.ndarray) -> str:
        dec = np.full((1, self.max_len), self.pad_id, dtype=np.int32)
        dec[0, 0] = self.bos_id

        for t in range(self.max_len - 1):
            probs = model.predict([src, dec], verbose=0)          # (1, T, V)
            nxt = int(np.argmax(probs[0, t, :]))
            dec[0, t + 1] = nxt
            if nxt == self.eos_id:
                break

        out_ids = [int(i) for i in dec[0] if int(i) not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode_ids(out_ids)

    def translate_en2tr(self, text: str) -> str:
        src = self._encode_src(text)
        return self._greedy_decode(self.en2tr, src)

    def translate_tr2en(self, text: str) -> str:
        src = self._encode_src(text)
        return self._greedy_decode(self.tr2en, src)

    def translate_auto(self, text: str, lang: str) -> str:
        """
        lang: 'eng' -> en2tr, 'tur' -> tr2en
        """
        if lang == "eng":
            return self.translate_en2tr(text)
        elif lang == "tur":
            return self.translate_tr2en(text)
        else:
            raise ValueError("lang must be 'eng' or 'tur'")

_TRANSLATOR = None

def get_translator() -> Seq2SeqTranslator:
    global _TRANSLATOR
    if _TRANSLATOR is None:
        _TRANSLATOR = Seq2SeqTranslator(models_dir="models")
    return _TRANSLATOR


def translate_en2tr(text: str) -> str:
    return get_translator().translate_en2tr(text)

def translate_tr2en(text: str) -> str:
    return get_translator().translate_tr2en(text)

def translate_auto(text: str, lang: str) -> str:
    return get_translator().translate_auto(text, lang)

