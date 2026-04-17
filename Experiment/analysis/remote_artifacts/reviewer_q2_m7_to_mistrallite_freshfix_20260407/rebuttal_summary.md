# Q2 Fresh Rerun Summary

## Calibration

| item | value |
|---|---:|
| calibration_samples | 200 |
| prefix_tokens_mean | 256.00 |
| prefix_tokens_min | 256 |
| prefix_tokens_max | 256 |
| total_calibration_prefix_tokens | 51200 |
| pairs_dir_size_bytes | 67126066950 |

## Offline Stages

| stage | success | wall_clock_sec | peak_gpu_memory_mb | max_cpu_rss_kb | gpu_ids |
|---|---|---:|---:|---:|---|
| capture_pair | True | 401.184 | 14997.0 | 15193308 | 0,1 |
| train_reuse | True | 383.051 | 1179.0 | 1307176 | 2 |
| profile | True | 828.255 | 7479.0 | 1457980 | 2 |
| build_route | True | 4.369 |  | 595432 |  |
| train_patch | True | 647.622 | 767.0 | 43118360 | 2 |

## Online Metrics

| mode | F1 | EM | TTFT_ms | total_ms | payload_mb | compression_ratio | encode_ms | transfer_ms | rebuild_ms | patch_infer_ms | recompute_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_consumer | 0.063023 | 0.000000 | 47.842 | 527.969 | 0.000 |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| reuse_only | 0.010224 | 0.000000 | 45.019 | 701.622 | 7.969 | 4.0 | 10.540 | 2.364 | 10.593 | 0.000 | 0.000 |
| reuse_fallback | 0.047237 | 0.000000 | 64.485 | 454.485 | 2.988 | 10.666666666666666 | 3.821 | 0.854 | 3.813 | 0.000 | 35.162 |
| reuse_patch | 0.025270 | 0.000000 | 68.335 | 708.907 | 14.941 | 2.1333333333333333 | 3.821 | 0.856 | 3.796 | 12.248 | 26.602 |
| reuse_patch_recompute | 0.030227 | 0.000000 | 74.992 | 716.718 | 14.941 | 2.1333333333333333 | 3.818 | 0.854 | 3.795 | 12.144 | 33.535 |

