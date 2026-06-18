# Qwen32 official-vLLM Full Consumer baselines

Scope: Qwen2.5-32B-Instruct Full Consumer baselines through official vLLM TP4. These rows are not SCD reuse/patch results.

| Dataset | n | Primary | F1 | EM | NLL | PPL | Output tok/s | Source |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cmrc2018 | 3219 | f1_em | 0.9098 | 0.7086 |  |  | 107.7083 | `Experiment/results/a800_qwen32_vllm_cmrc_full_tp4_chat_20260618/summary.json` |
| hotpotqa | 7405 | f1_em | 0.7459 | 0.5857 |  |  | 29.5019 | `Experiment/results/a800_qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618/hotpotqa_full/summary.json` |
| wikitext2 | 763 | ppl | 0.2717 | 0.0000 | 1.9711 | 7.1788 | 429.7665 | `Experiment/results/a800_qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618/wikitext2_full/summary.json` |
