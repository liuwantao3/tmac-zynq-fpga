#!/usr/bin/env python3
"""Qwen2-0.5B chat via INT16 FPGA AXI simulation.

Usage:
  echo "Hello, how are you?" | python3 sim/chat.py                  # single turn
  python3 sim/chat.py                                               # interactive
  python3 sim/chat.py --fpga-int16                                  # force INT16
"""

import sys, os, json, subprocess, readline

SIMDIR = os.path.dirname(os.path.abspath(__file__))
PROJDIR = os.path.dirname(SIMDIR)
BMODEL = os.path.join(PROJDIR, "models", "model.tmac")
BINARY = os.path.join(SIMDIR, "tmac_gguf")
VOCAB  = os.path.join(SIMDIR, "vocab.json")
MERGES = os.path.join(SIMDIR, "merges.txt")

MAX_GEN = 32

class Qwen2Chat:
    def __init__(self, fpga_int16=False):
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer(vocab_file=VOCAB, merges_file=MERGES)
        special = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(special.keys())})
        self.tokenizer.add_tokens(list(special.keys()))
        self.im_start = 151644
        self.im_end   = 151645
        self.eos      = 151643
        self.fpga_flag = ["--fpga-int16"] if fpga_int16 else []
        self.history = []

    def build_prompt(self, user_text):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for role, text in self.history:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_text})
        prompt = ""
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def generate(self, user_text):
        prompt = self.build_prompt(user_text)
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > 256 - MAX_GEN:
            self.history = []
            prompt = self.build_prompt(user_text)
            tokens = self.tokenizer.encode(prompt)

        token_str = "\n".join(str(t) for t in tokens)
        try:
            proc = subprocess.run(
                [BINARY, BMODEL, "--generate", str(MAX_GEN)] + self.fpga_flag,
                input=token_str, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return " [timeout]"

        out_tokens = []
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line and line.isdigit():
                tid = int(line)
                if tid == self.eos or tid == self.im_end:
                    break
                out_tokens.append(tid)

        response = self.tokenizer.decode(out_tokens)
        for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
            response = response.replace(tok, "")
        response = response.strip()
        self.history.append(("user", user_text))
        self.history.append(("assistant", response))
        return response


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2-0.5B chat")
    parser.add_argument("--fpga-int16", action="store_true", help="Use INT16 FPGA path")
    parser.add_argument("prompt", nargs="?", help="Single prompt (reads from stdin if omitted)")
    args = parser.parse_args()

    if not os.path.exists(VOCAB) or not os.path.exists(MERGES):
        print("[ERROR] Run extract_tokens.py first", file=sys.stderr)
        sys.exit(1)

    bot = Qwen2Chat(fpga_int16=args.fpga_int16)

    if args.prompt:
        resp = bot.generate(args.prompt)
        print(resp)
        return

    if not sys.stdin.isatty():
        prompt_text = sys.stdin.read().strip()
        if prompt_text:
            resp = bot.generate(prompt_text)
            sys.stdout.write(resp)
            sys.stdout.write("\n")
        return

    print("Qwen2-0.5B chat (INT16 FPGA sim). Ctrl+D to exit.\n")
    while True:
        try:
            user = input(">>> ").strip()
            if not user:
                continue
            if user.lower() in ("/exit", "/quit"):
                break
            if user.lower() == "/reset":
                bot.history = []
                print("[reset]")
                continue
            resp = bot.generate(user)
            print(resp)
            print()
        except (EOFError, KeyboardInterrupt):
            break

    print()

if __name__ == "__main__":
    main()
