# Data Preparation for Strict SCD-ICML Reproduction

Set `DATA_ROOT` to a directory containing the benchmark files below. The loader
also checks `DATA_ROOT/data`, so either layout is acceptable.

## Required Files

CMRC2018:

```text
DATA_ROOT/
  cmrc2018/
    cmrc2018_train.json
    cmrc2018_dev.json
```

HotpotQA:

```text
DATA_ROOT/
  hotpotqa/
    hotpot_dev_distractor_v1.json
```

Accepted alternatives include:

```text
DATA_ROOT/hotpotqa_official/hotpot_dev_distractor_v1.json
DATA_ROOT/hotpotqa_official/hotpot_dev_distractor_v1.full.json
DATA_ROOT/tmp/hotpotqa_official/hotpot_dev_distractor_v1.json
DATA_ROOT/tmp/hotpotqa_official/hotpot_dev_distractor_v1.full.json
DATA_ROOT/HotpotQA/raw/hotpot_dev_distractor_v1.json
DATA_ROOT/droidspeak/hotpotqa.json
```

WikiText-2:

```text
DATA_ROOT/
  wikitext-2/
    wiki.test.txt
```

Accepted alternatives include:

```text
DATA_ROOT/WikiText-2/wiki.test.txt
DATA_ROOT/wikitext/wiki.test.txt
```

## Model Config

Copy the template and edit the two local model paths:

```bash
cp configs/models_mistrallite_to_mistral7_template.json configs/models_local_mistrallite_to_mistral7.json
```

Then run the wrapper with:

```bash
MODELS_CONFIG_LITE_TO_7B=configs/models_local_mistrallite_to_mistral7.json \
DATA_ROOT=/absolute/path/to/data-root \
bash scripts/run_ser_scd_icml_mistral_repro.sh
```

The verified `ser` run used:

```text
DATA_ROOT=/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854
MODELS_CONFIG_LITE_TO_7B=configs/models_ser_mistrallite_to_mistral7_mql.json
```

Those local paths are included for reproducibility evidence, not as portable
defaults for other machines.
