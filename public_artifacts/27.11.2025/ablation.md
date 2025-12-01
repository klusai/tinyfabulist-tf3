# Ablation Benchmark Summary

- Baseline loss: 5.5676
- Baseline perplexity: 261.80

| MLP Rate | Attn Rate | Abl Loss | Abl PPL | ΔLoss% | ΔPPL% |
|--------:|----------:|---------:|--------:|-------:|------:|
| 0.10 | 0.10 | 5.6576 | 286.45 | <span style="background-color:#fef7f6;padding:0 6px;border-radius:4px;">+1.62%</span> | <span style="background-color:#fad2cf;padding:0 6px;border-radius:4px;">+9.42%</span> |
| 0.10 | 0.30 | 6.1342 | 461.37 | <span style="background-color:#f9cfcb;padding:0 6px;border-radius:4px;">+10.18%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+76.23%</span> |
| 0.10 | 0.50 | 6.3448 | 569.51 | <span style="background-color:#f7bdb8;padding:0 6px;border-radius:4px;">+13.96%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+117.54%</span> |
| 0.10 | 0.70 | 7.5468 | 1894.64 | <span style="background-color:#ec574b;padding:0 6px;border-radius:4px;">+35.55%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+623.71%</span> |
| 0.30 | 0.10 | 6.0512 | 424.61 | <span style="background-color:#fad6d3;padding:0 6px;border-radius:4px;">+8.69%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+62.19%</span> |
| 0.30 | 0.30 | 6.3869 | 594.01 | <span style="background-color:#f7b9b4;padding:0 6px;border-radius:4px;">+14.72%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+126.90%</span> |
| 0.30 | 0.50 | 7.0261 | 1125.63 | <span style="background-color:#f1837a;padding:0 6px;border-radius:4px;">+26.20%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+329.97%</span> |
| 0.30 | 0.70 | 6.8095 | 906.46 | <span style="background-color:#f3968e;padding:0 6px;border-radius:4px;">+22.31%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+246.25%</span> |
| 0.50 | 0.10 | 7.3884 | 1617.15 | <span style="background-color:#ed6559;padding:0 6px;border-radius:4px;">+32.70%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+517.71%</span> |
| 0.50 | 0.30 | 7.2977 | 1476.97 | <span style="background-color:#ee6c62;padding:0 6px;border-radius:4px;">+31.08%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+464.17%</span> |
| 0.50 | 0.50 | 7.7047 | 2218.83 | <span style="background-color:#ea4a3d;padding:0 6px;border-radius:4px;">+38.39%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+747.54%</span> |
| 0.50 | 0.70 | 8.7249 | 6154.27 | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+56.71%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+2250.79%</span> |
| 0.70 | 0.10 | 7.5440 | 1889.41 | <span style="background-color:#ec584b;padding:0 6px;border-radius:4px;">+35.50%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+621.71%</span> |
| 0.70 | 0.30 | 8.5890 | 5372.18 | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+54.27%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+1952.05%</span> |
| 0.70 | 0.50 | 9.0464 | 8487.59 | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+62.48%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+3142.06%</span> |
| 0.70 | 0.70 | 8.2265 | 3738.63 | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+47.76%</span> | <span style="background-color:#ea4335;padding:0 6px;border-radius:4px;">+1328.07%</span> |
