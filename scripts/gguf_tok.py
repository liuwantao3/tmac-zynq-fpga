#!/usr/bin/env python3
"""Qwen2 GGUF tokenizer: encode (text->ids) + decode (ids->text), pure Python.

Reads the tokenizer metadata directly from the GGUF (no external deps, no
`transformers`/`tiktoken`/`gguf` packages needed). Model is byte-level BPE
(tokenizer.ggml.model == "gpt2").

Usage:
  python3 scripts/gguf_tok.py --decode 9370 3837 35946          # ids -> text
  python3 scripts/gguf_tok.py --encode "Hello!"                 # text -> ids
  python3 scripts/gguf_tok.py --chat "Hello! How are you?"      # chat-template prompt -> ids
  python3 scripts/gguf_tok.py --run [--greedy] [--max N]        # run sim, decode output to words

Feeding a real prompt (not token 151646, which is a PAD token) is what makes the
benchmark output readable. Use --chat to build the Qwen2 chat prompt, then pass
the printed ids to tmac (board) or the sim (host). Decode board/sim token output
with --decode.
"""

import struct
import sys
import re
import os
import subprocess


def read_string(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n)


def read_kv(f):
    """Read one GGUF key-value pair. Returns (key, value)."""
    key = read_string(f).decode('utf-8', 'replace')
    t = struct.unpack('<I', f.read(4))[0]
    if t == 0:   v = struct.unpack('<B', f.read(1))[0]
    elif t == 1: v = struct.unpack('<b', f.read(1))[0]
    elif t == 2: v = struct.unpack('<H', f.read(2))[0]
    elif t == 3: v = struct.unpack('<h', f.read(2))[0]
    elif t == 4: v = struct.unpack('<I', f.read(4))[0]
    elif t == 5: v = struct.unpack('<i', f.read(4))[0]
    elif t == 6: v = struct.unpack('<f', f.read(4))[0]
    elif t == 7: v = struct.unpack('<?', f.read(1))[0]
    elif t == 8: v = read_string(f)
    elif t == 9:
        etype = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        arr = []
        for _ in range(n):
            if etype == 8: arr.append(read_string(f).decode('utf-8', 'replace'))
            elif etype == 0: arr.append(struct.unpack('<B', f.read(1))[0])
            elif etype == 1: arr.append(struct.unpack('<b', f.read(1))[0])
            elif etype == 4: arr.append(struct.unpack('<I', f.read(4))[0])
            elif etype == 5: arr.append(struct.unpack('<i', f.read(4))[0])
            elif etype == 6: arr.append(struct.unpack('<f', f.read(4))[0])
            elif etype == 10: arr.append(struct.unpack('<Q', f.read(8))[0])
            elif etype == 11: arr.append(struct.unpack('<q', f.read(8))[0])
            elif etype == 12: arr.append(struct.unpack('<d', f.read(8))[0])
            else:
                raise ValueError('unsupported array elem type %d' % etype)
        v = arr
    elif t == 10: v = struct.unpack('<Q', f.read(8))[0]
    elif t == 11: v = struct.unpack('<q', f.read(8))[0]
    elif t == 12: v = struct.unpack('<d', f.read(8))[0]
    else:
        raise ValueError('unsupported type %d' % t)
    return key, v


def load_vocab(path):
    f = open(path, 'rb')
    magic = f.read(4)
    assert magic == b'GGUF', 'bad magic %r' % magic
    f.read(4)  # version
    f.read(8)  # n_tensors
    n_kv = struct.unpack('<Q', f.read(8))[0]
    meta = {}
    for _ in range(n_kv):
        k, v = read_kv(f)
        meta[k] = v
    f.close()
    return meta


def _tiktoken_byte_encoder():
    """tiktoken byte->unicode map (as used by GPT-2 style byte-level BPE)."""
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('\xa1'), ord('\xac') + 1)) + \
         list(range(ord('\xae'), ord('\xff') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENC = _tiktoken_byte_encoder()
_BYTE_DEC = {c: b for b, c in _BYTE_ENC.items()}


def _bytes_to_tokens(tok):
    """GGUF token string (byte-level BPE chars) -> raw bytes."""
    return bytes(_BYTE_DEC.get(ch, ord(ch)) for ch in tok)


class Tokenizer:
    def __init__(self, path):
        meta = load_vocab(path)
        self.tokens = meta['tokenizer.ggml.tokens']
        self.merges = meta['tokenizer.ggml.merges']
        self.token_type = meta.get('tokenizer.ggml.token_type',
                                   [0] * len(self.tokens))
        self.bos = int(meta.get('tokenizer.ggml.bos_token_id', 0))
        self.eos = int(meta.get('tokenizer.ggml.eos_token_id', 0))
        self._id_of = {}
        self._id_bytes = {}
        for i, t in enumerate(self.tokens):
            self._id_of[t] = i
            self._id_bytes[i] = _bytes_to_tokens(t)
        # 1=normal, 2=unknown, 3=control, 4=user-defined
        self.special = {i: t for i, t in enumerate(self.tokens)
                        if self.token_type[i] in (3, 4)}
        self._merge_rank = {}
        for rank, m in enumerate(self.merges):
            self._merge_rank[m] = rank
        pat_parts = [re.escape(t) for t in self.special.values() if t]
        self.special_re = re.compile('(' + '|'.join(pat_parts) + ')') \
            if pat_parts else None

    def encode(self, text):
        """text -> token ids using byte-level BPE."""
        ids = []
        if self.special_re:
            for piece in self.special_re.split(text):
                if not piece:
                    continue
                if piece in self._id_of and \
                        self.token_type[self._id_of[piece]] in (3, 4):
                    ids.append(self._id_of[piece])
                else:
                    ids.extend(self._encode_piece(piece))
        else:
            ids.extend(self._encode_piece(text))
        return ids

    def _encode_piece(self, text):
        data = text.encode('utf-8')
        toks = [_BYTE_ENC[b] for b in data]
        if len(toks) <= 1:
            j = ''.join(toks)
            return [self._id_of[j]] if j in self._id_of else []
        while len(toks) > 1:
            best_rank = None
            best_pair = None
            for i in range(len(toks) - 1):
                pair = toks[i] + ' ' + toks[i + 1]
                if pair in self._merge_rank:
                    r = self._merge_rank[pair]
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_pair = i
            if best_pair is None:
                break
            merged = toks[best_pair] + toks[best_pair + 1]
            toks = toks[:best_pair] + [merged] + toks[best_pair + 2:]
        out = []
        for t in toks:
            if t in self._id_of:
                out.append(self._id_of[t])
        return out

    def decode(self, ids, skip_special=True):
        parts = []
        for i in ids:
            if i < 0 or i >= len(self.tokens):
                parts.append(('<OOB:%d>' % i).encode())
                continue
            typ = self.token_type[i]
            if typ == 3:
                if skip_special:
                    continue
                parts.append(('<' + self.tokens[i].strip('<>') + '>').encode())
                continue
            parts.append(self._id_bytes[i])
        raw = b''.join(parts)
        return raw.decode('utf-8', 'replace')


def chat_prompt(text):
    return ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
            '<|im_start|>user\n' + text + '<|im_end|>\n'
            '<|im_start|>assistant\n')


def run_sim(tok, max_tokens, greedy, extra=(), prompt_ids=None):
    """Run the host sim, decode its generated token IDs to words."""
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    binary = os.path.join(proj, 'sim', 'tmac_gguf')
    model = os.path.join(proj, 'models', 'model.tmac')
    if not os.path.exists(binary):
        print('[ERROR] Build sim first: g++ -std=c++17 -pthread -O2 -I sim -I gguf -I . sim/tmac_gguf.cpp sim/matmul_q8.cpp -o sim/tmac_gguf',
              file=sys.stderr)
        return
    args = [binary, model, '--generate', str(max_tokens)]
    if greedy:
        args.append('--greedy')
    args.extend(extra)
    if prompt_ids is None:
        prompt_ids = [tok.bos] if tok.bos else [151646]
    stdin_data = '\n'.join(str(i) for i in prompt_ids) + '\n'
    proc = subprocess.run(args, input=stdin_data, capture_output=True, text=True)
    out = []
    for line in proc.stdout.strip().split('\n'):
        line = line.strip()
        if line and line.isdigit():
            out.append(int(line))
    print('prompt ids :', ' '.join(str(i) for i in prompt_ids))
    print('prompt text:', repr(tok.decode(prompt_ids)))
    print('gen ids   :', ' '.join(str(i) for i in out))
    print('gen text  :', tok.decode(out))


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Qwen2 GGUF tokenizer (encode/decode)')
    ap.add_argument('--gguf', default=None,
                    help='path to Qwen2 GGUF (default: models/qwen2-0_5b-instruct-q4_k_m.gguf)')
    ap.add_argument('--decode', nargs='+', type=int, help='token ids -> text')
    ap.add_argument('--encode', nargs='+', help='text -> token ids')
    ap.add_argument('--chat', help='chat-prompt text -> token ids')
    ap.add_argument('--run', action='store_true', help='run host sim, decode output to words')
    ap.add_argument('--prompt', nargs='+', default=None,
                    help='prompt text for --run (default: single BOS)')
    ap.add_argument('--max', type=int, default=32, help='max generated tokens (--run)')
    ap.add_argument('--greedy', action='store_true', help='greedy sampling (--run)')
    ap.add_argument('--fpga', nargs='*', default=None,
                    help='sim FPGA path flags, e.g. --fpga --fpga-q8 --fpga-q5-0')
    args = ap.parse_args()

    gguf = args.gguf
    if gguf is None:
        gguf = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'models',
            'qwen2-0_5b-instruct-q4_k_m.gguf')
    tok = Tokenizer(gguf)

    if args.decode:
        print(tok.decode(args.decode))
    elif args.encode:
        print(' '.join(str(i) for i in tok.encode(' '.join(args.encode))))
    elif args.run:
        pid = None
        if args.prompt:
            pid = tok.encode(' '.join(args.prompt))
        elif args.chat:
            pid = tok.encode(chat_prompt(args.chat))
        run_sim(tok, args.max, args.greedy,
                tuple(args.fpga) if args.fpga else (), prompt_ids=pid)
    elif args.chat:
        print(' '.join(str(i) for i in tok.encode(chat_prompt(args.chat))))
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
