from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Mistral prefill diagnostic.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="请简要解释语义缓存蒸馏。")
    args = parser.parse_args()

    for dtype_name, torch_dtype in [("bfloat16", torch.bfloat16), ("float16", torch.float16), ("float32", torch.float32)]:
        try:
            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map={"": args.device},
            )
            model.eval()
            enc = tok(args.prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            print(dtype_name, "OK", tuple(out.logits.shape))
            del out, model, tok, input_ids, attention_mask
            torch.cuda.empty_cache()
        except Exception as exc:
            print(dtype_name, "FAIL", repr(exc))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
