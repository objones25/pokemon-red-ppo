# Performance bottlenecks for PPO on PyBoy (Pokémon Red)

> Some claims here were later corrected; see `docs/CONFLICT_RESOLUTIONS.md`.

Measured 2026-11-03 on an Apple M4 Pro (10 performance + 4 efficiency cores, 24 GB) with PyBoy 2.9.0, PyTorch 2.15.0 and Python 3.12. `MPS` is PyTorch's Apple GPU backend.

## Summary

**The emulator limits training, not the neural network, and FLOPs are the wrong way to measure it.** PyBoy does almost no floating-point work, yet one 24-frame environment step uses about 16 times the CPU cycles of a Nature CNN forward pass.

| Limit | Measured evidence | When it applies |
|---|---|---|
| **1. Emulator CPU time** | About 2.4 ms per step, or about 370–460 steps/s per core. This Mac tops out at 5,155 steps/s with no policy attached. | Always. Sets the throughput ceiling. |
| **2. Lockstep overhead between processes** | Adds 1.6 ms per step at 14 workers. Raw throughput falls from 5,155 to about 3,300 steps/s. | One env per process. Reduced by running several envs per worker. |
| **3. Small-batch inference** | Takes 33–35% of wall time with 12 envs on MPS, falling to 15–17% with 96 envs. | Few envs per inference batch. |
| **4. Gradient updates** | 15% of wall time for a Nature CNN, 45–48% for IMPALA, 79% for IMPALA ×4. | Large models or many epochs. |
| **5. Samples needed** | Published runs used up to 400M steps for about 20% of the game. | Always. Decides whether a run is feasible. |

On this Mac, the full loop went from about 1,590 to about 2,510 steps/s (+58%) by running 8 envs per worker and using a cheaper observation path. At 2,500 steps/s, 400M steps takes about 44 hours and 10B steps about 46 days.

For GPU servers, see [Scaling on GPU machines](#scaling-on-gpu-machines). In short, PyBoy stays on the CPU, so **a GPU only speeds up the policy, and throughput depends on the balance of CPU cores to GPU speed**.

## FLOPs for data collection

FLOP values come from `torch.utils.flop_counter`, which counts a multiply-add as 2 FLOPs. I checked this against a hand count of one convolution and the source (`flop_counter.py:301`). The input is 3 stacked 72×80 frames.

### Cost of one rollout step

Instruction and cycle counts are per call, from `/usr/bin/time -l` with startup cost subtracted. Rates and shares below are for a single CPU thread at batch size 1 unless stated.

| Component | FLOPs | Instructions | CPU cycles | Cycles vs emulator |
|---|---|---|---|---|
| **PyBoy, 24 frames** | **≈0 (see below)** | **40.26M** | **7.54M** | **100%** |
| Observation (float greyscale + downsample) | ≈115k (hand count) | About 4.3M with RAM reads | About 0.65M with RAM reads | 9% |
| Nature CNN forward | 11.8M | 2.42M | 0.47M | 6% |
| Nature CNN + LSTM-512 forward | 16.0M | 3.03M | 0.71M | 9% |
| IMPALA ×1 forward | 86.1M | 19.18M | 4.12M | 55% |
| IMPALA ×4 forward | 1,299.8M | 86.13M | 21.45M | 284% |

A rollout step therefore costs 11.8M FLOPs for a Nature CNN, plus a small amount for the observation. FLOPs miss the largest cost because emulation is integer work:

- **Emulator FLOPs:** with sound off, the emulator core has almost no floating-point arithmetic. A static scan of the compiled code found 1 such instruction in `cpu`, 2 in `mb`, and 0 in `lcd`, `opcodes`, `ram` and `timer`; `sound` has 41. This is a scan of the binary, not a runtime count, so "≈0" is strongly implied rather than directly measured.
- **Emulation workload:** one step is 24 frames × 70,224 Game Boy clock cycles (hardware spec), or 1.69M guest cycles. PyBoy uses about 24 host instructions and 4.5 host cycles per guest cycle.
- **Opportunity cost:** a CPU core delivers about 21–25 FLOPs per cycle on the three smaller models. The cycles for one emulator step could instead do about 160–190M FLOPs of CNN work, which is 13–16 Nature CNN forward passes.
- **Inefficient NumPy:** the float observation path is only about 1% of the policy's FLOPs but took 172 µs, more than a whole CNN forward pass per observation at batch 12 (110 µs). That is about 0.67 GFLOP/s, against 65–210 GFLOP/s for the networks on one CPU thread.

### FLOPs per frame across training

This uses the accounting `FPF = F_fwd × (1 + 3 × passes)` at 3 PPO epochs, assuming a backward pass costs twice a forward pass. Rollout collection is only 10% of total FLOPs.

| Model | Forward FLOPs | FPF | Total at 400M steps | Total at 10B steps |
|---|---|---|---|---|
| Nature CNN | 11.8M | 118M | 4.7×10¹⁶ | 1.2×10¹⁸ |
| Nature + LSTM | 16.0M | 160M | 6.4×10¹⁶ | 1.6×10¹⁸ |
| IMPALA ×1 | 86.1M | 861M | 3.4×10¹⁷ | 8.6×10¹⁸ |
| IMPALA ×4 | 1,299.8M | 13.0G | 5.2×10¹⁸ | 1.3×10²⁰ |

### Why FLOPs don't predict training time

The highest FLOP rate I measured on the GPU was 5.1 TFLOP/s (IMPALA ×4 inference, batch 64). This table compares that with what full training actually demanded. Throughputs are from the full-loop runs below, and all but the first row use the 12-env setup.

| Model | Steps/s | FLOPs demanded | Share of 5.1 TFLOP/s |
|---|---|---|---|
| Nature CNN, tuned loop | 2,500 | 0.30 TFLOP/s | 6% |
| Nature + LSTM | 1,371 | 0.22 TFLOP/s | 4% |
| IMPALA ×1 | 874 | 0.75 TFLOP/s | 15% |
| IMPALA ×4 | 267 | 3.47 TFLOP/s | 68% |

With a Nature CNN, the GPU spends most of its capacity idle while the CPU is stuck emulating. A FLOP-based estimate puts 400M Nature CNN steps at about 2.6 hours at that peak rate. The realistic figure is about 44 hours. FLOPs only start predicting wall time for large models such as IMPALA ×4.

## Where the time goes

### Inside one emulator step

| Operation | Cost | Takeaway |
|---|---|---|
| 24 frames, random actions, Pallet Town | 2,367 µs | The main cost |
| Rendering the last frame | About 19 µs (1%) | Not worth optimizing further |
| Rendering every frame | 81 vs 70 µs per frame | Rendering only the last frame already avoids this |
| Sound emulation on | +212 µs (+9%) in Pallet Town, +54 µs (+3%) in Red's room | Pass `sound_emulated=False` |
| Observation via float mean | 172 µs | Avoidable, see below |
| Observation via slice `[::2,::2,0]` | 1.8 µs | About 100× cheaper, but changes the image |
| 41 RAM reads | 0.73 µs | Negligible, so RAM-based rewards are cheap |
| `game_area()` | 2.4 µs | Negligible |
| `load_state` | 2.9 ms | About one step, negligible per episode |
| `save_state` | 8.9 ms | About 3.6 steps, which matters if you archive states often |
| New PyBoy instance | 11.7 ms | Negligible |

Raw emulation peaks at about 15,100 frames/s, about 250× real time.

**Cost depends on game state** by almost a factor of 2, and I didn't reach a battle:

| State | µs per 24 frames |
|---|---|
| Pallet Town, idle | 1,545–1,550 |
| Red's room, random actions | 1,564–1,585 |
| Pallet Town, walking left and right | 1,612–1,613 |
| Pallet Town, random actions | 2,389–2,396 |
| Red's room, idle | 2,636–2,683 |
| Start menu open, idle | 2,765–2,771 |

### Scaling with parallel processes

"Free-running" workers step independently with no policy. "Lockstep" workers wait for each other at every step, as PPO's synchronous rollout does, but no network runs.

| Processes | Free-running steps/s | Lockstep steps/s | Sync overhead per step |
|---|---|---|---|
| 1 | 380 | 393–396 | 0.02 ms |
| 4 | 1,850 | 1,370–1,395 | 0.63–0.65 ms |
| 8 | 3,660 | 2,503–2,518 | 0.94–0.95 ms |
| 10 | 4,389 | 2,955–2,964 | 1.05–1.06 ms |
| 12 | 4,759 | 3,005–3,024 | 1.47–1.48 ms |
| 14 | 5,155 | 3,312–3,352 | 1.59–1.63 ms |

Free-running values are from pass 2. Pass 1 gave 4,187, 4,507 and 4,907 at 10, 12 and 14 processes. More processes than cores won't help: in the contaminated first run, going from 14 to 28 processes gained under 5%.

### Full PPO loop

Rollout length T=128, 3 epochs, minibatch 512, Nature CNN on the MPS update unless noted. Each range is two runs. "Env %" is time waiting on the emulators.

| Workers × envs per worker | Inference | Obs path | Steps/s | Env % | Inference % | Update % |
|---|---|---|---|---|---|---|
| 12 × 1 | MPS | Float | 1,586–1,595 | 50–52 | 33–35 | 14–15 |
| 12 × 1 | MPS | Slice | 1,626–1,784 | 48–50 | 35–36 | 15 |
| 12 × 1 | CPU | Float | 1,951–1,961 | 62–63 | 18–19 | 18–19 |
| 12 × 1 | CPU | Slice | 2,033–2,161 | 59–63 | 18–20 | 18–20 |
| 12 × 4 | CPU | Slice | 1,623–1,767 | 49 | 36 | 14–15 |
| 12 × 4 | MPS | Slice | 2,167–2,208 | 62–64 | 17–18 | 19–20 |
| 14 × 4 | MPS | Slice | 2,312–2,313 | 60–63 | 17–20 | 20 |
| **12 × 8** | **MPS** | **Slice** | **2,484–2,537** | **61–64** | **15–17** | **21–22** |

- **Inference device depends on batch size.** At 12 envs, CPU beats MPS by 23%. At 48 envs, MPS wins by about 30%.
- **Update on CPU is not viable.** With both inference and update on the CPU, throughput was 696–717 steps/s, with about 70% of the time in the update.

Model size and epochs, with 12 × 1 workers, MPS and the float observation path. These come from an earlier version of the loop script that gave PyTorch 1 CPU thread instead of 4, so its Nature baseline reads about 10% lower than the table above (1,422–1,446 vs 1,586–1,595). Compare within each table, not across them.

| Configuration | Steps/s | Env % | Inference % | Update % |
|---|---|---|---|---|
| Nature CNN | 1,422–1,446 | 51–53 | 32 | 15–16 |
| Nature + LSTM | 1,371–1,390 | 48–49 | 34 | 16–17 |
| IMPALA ×1 | 874–921 | 29–32 | 22–23 | 45–48 |
| IMPALA ×4 | 263–267 | 8–9 | 12 | 79 |
| Nature, 10 epochs, minibatch 256 | 1,026–1,027 | 31–33 | 23–24 | 43–46 |
| Nature, T=2048 | 1,441–1,530 | 47–50 | 35–36 | 15–16 |

### When the update takes over

Update cost on MPS per sample per pass: Nature 21 µs, LSTM 25 µs, IMPALA ×1 159 µs, IMPALA ×4 970 µs.

At 2,500 steps/s, collecting one sample takes 400 µs. The update overtakes collection at about 19 passes for Nature, 16 for LSTM, 2.5 for IMPALA ×1 and 0.4 for IMPALA ×4. The LSTM figure uses a sequence length of 1, so real truncated backpropagation through time will cost more.

## Limits beyond throughput

### Samples needed

| Source | Setup | Result |
|---|---|---|
| [Pleines et al. 2025](https://arxiv.org/abs/2502.19920) | 32 workers × 2,048 steps = 65,536 batch; 24 frames per decision | Figures plot up to 400M steps; goal was about 20% of the game, through Cerulean; about 36 h per run |
| [`pokemonred_puffer` config](https://github.com/drubinstein/pokemonred_puffer) | 288 envs, 24 workers, 65,536 batch, 3 epochs | `total_timesteps: 10_000_000_000`, with a comment of 100B "for full games" |

The paper reports about 392 steps/s at 24 frames per decision on a Ryzen 7 2700X, in line with the 370–460 per process I measured. If its runs do last 400M steps, 36 h implies about 3,100 steps/s, similar to what this Mac can reach. That 400M figure comes from the plot axes, so treat the run length as approximate.

Wall-clock budget, using the tuned rate, the untuned rate, and the policy-free ceiling:

| Steps | Game time | 1,446 steps/s | 2,500 steps/s | 5,155 steps/s (ceiling) |
|---|---|---|---|---|
| 400M | About 44,600 h | 77 h | 44 h | 22 h |
| 10B | About 1.1M h | 80 days | 46 days | 22 days |
| 100B | About 11M h | 2.2 years | 1.3 years | 225 days |

A 400M-step run is feasible on this Mac, while a 10B run is impractical and 100B is out of reach. Reaching those needs a machine with many more CPU cores.

### Other limits

- **Weak signal per step.** From 30,000 random steps per start state, 23–27% of steps left the screen identical and only about 40% changed the player's position. A random policy found only 417–488 unique positions across 5–6 maps. This is a random policy in the early game, so it shows how sparse progress is rather than how PPO would do.
- **Slow evaluation.** Pleines et al. report that evaluation episodes take as much wall time as training itself. A 19,816-step episode (the puffer `max_steps`) is about 2.2 h of game time, or about 13 minutes per env at the 96-env rate here.
- **Long memory.** Pleines et al. report that their GRU run needed CPU only because of the VRAM cost of long sequences, averaging 24 days per run.
- **RAM.** Each PyBoy env uses about 71 MB, so 96 envs is about 6.8 GB. A rollout buffer of stacked frames is 0.42 GB for 12 × 2,048 and 1.13 GB for 32 × 2,048. Neither is a limit on 24 GB at these sizes.

## Scaling on GPU machines

**This section is a projection, not a measurement.** No CUDA GPU or server CPU was tested. The model below is fitted to the Mac runs, and two of its inputs are assumptions.

### The model

Each synchronous lockstep advances W workers × k envs per worker:

```
time per step = (k·b + O(W) + L) / (W·k)   ← shrinks as cores are added
              + c                          ← serial, main process
              + U / g                      ← serial, GPU update
```

| Term | Meaning | Value | Source |
|---|---|---|---|
| `b` | Emulator time per step | 2.6 ms on the M4 Pro; **5 ms per cloud vCPU** | Measured; **assumed** (vCPUs are usually hyperthreads) |
| `O(W)` | Lockstep sync overhead | 0.18 + 0.10·W ms | Fit to 4–14 workers, extrapolated beyond that |
| `L` | Inference latency per batch | 1.96 ms | Fit to full-loop runs on MPS |
| `c` | Per-observation inference pipeline | 43 µs | Fit to full-loop runs (includes CPU-side conversion and sampling) |
| `U` | Update time per step at 3 epochs | Nature 80 µs, IMPALA ×1 510 µs, IMPALA ×4 2,980 µs | Full-loop runs on MPS |
| `g` | GPU speed relative to the M4 Pro | **What-if values: 1, 5, 20** | **Assumed.** No CUDA GPU has been measured |

Model vs measured on the Mac. It runs 2–16% optimistic, so treat projections as upper bounds.

| Config | Predicted steps/s | Measured steps/s |
|---|---|---|
| 12 × 1, Nature | 1,618 | 1,586–1,595 |
| 12 × 4, Nature | 2,445 | 2,167–2,208 |
| 14 × 4, Nature | 2,691 | 2,312–2,313 |
| 12 × 8, Nature | 2,673 | 2,484–2,537 |
| 12 × 1, IMPALA ×1 | 954 | 874–921 |
| 12 × 1, IMPALA ×4 | 284 | 263–267 |

### Projected throughput

Steps/s at k=8, b=5 ms per vCPU. "Emulator only" is the ceiling with zero policy and sync cost.

| vCPUs | Emulator only | Nature, g=1 | Nature, g=5 | Nature, g=20 | IMPALA ×1, g=5 | IMPALA ×1, g=20 | IMPALA ×4, g=5 | IMPALA ×4, g=20 |
|---|---|---|---|---|---|---|---|---|
| 12 | 2,400 | 1,742 | 1,960 | 2,008 | 1,678 | 1,924 | 917 | 1,555 |
| 32 | 6,400 | 3,336 | 4,242 | 4,469 | 3,108 | 4,077 | 1,226 | 2,712 |
| 64 | 12,800 | 4,599 | 6,517 | 7,069 | 4,176 | 6,137 | 1,363 | 3,491 |
| 128 | 25,600 | 5,672 | 8,905 | 9,970 | 5,043 | 8,210 | 1,444 | 4,077 |
| 256 | 51,200 | 6,422 | 10,902 | 12,543 | 5,627 | 9,879 | 1,489 | 4,450 |
| 512 | 102,400 | 6,876 | 12,280 | 14,402 | 5,972 | 10,997 | 1,512 | 4,663 |

Three limits show up as the machine grows:

1. **Few cores: the emulator.** At 12 vCPUs with a Nature CNN, going from g=5 to g=20 adds only 2%, so a faster GPU is mostly wasted.
2. **Mid-size: lockstep sync.** The growing gap to "emulator only" comes largely from the `O(W)` extrapolation and from stopping all emulators during inference. At 128 vCPUs the projection reaches only about 39% of the emulator ceiling, even at g=20. Asynchronous env batching, as `pokemonred_puffer` configures (`env_batch_size: 36` of 288 envs), targets this directly but I didn't test it.
3. **Many cores: serial work.** `c + U/g` sets a hard ceiling no matter how many cores you add:

| Model | g=1 | g=5 | g=20 | g=100 |
|---|---|---|---|---|
| Nature | 8,158 | 17,070 | 21,467 | 23,051 |
| IMPALA ×1 | 1,810 | 6,916 | 14,688 | 20,972 |
| IMPALA ×4 | 331 | 1,566 | 5,220 | 13,815 |

Everything converges on about 23,000 steps/s because of `c`, the main-process pipeline in my benchmark. That is a code limit, not a hardware one. Sending `uint8` frames to the GPU instead of converting on the CPU should raise it, though I haven't measured by how much.

### Balancing CPU and GPU

The vCPU count where emulation and the GPU update take equal time is `b / (U/g)`. Below it, extra GPU speed is mostly wasted. Above it, extra cores are.

| Model | g=1 | g=5 | g=20 |
|---|---|---|---|
| Nature | 62 | 312 | 1,250 |
| IMPALA ×1 | 10 | 49 | 196 |
| IMPALA ×4 | 2 | 8 | 34 |

This ratio is the most useful number for choosing hardware. With a Nature CNN, a typical GPU pod has far more GPU than its CPUs can feed. IMPALA ×4 is the opposite case.

### What the catalog offers

From the Runpod catalog on 2026-11-03. `Min vCPUs` is the listed floor per GPU, and a specific machine may offer more. Raw steps/s is vCPUs ÷ `b`, from 5 ms to 2.6 ms per step, before any overhead, so the cost range is a lower bound. Most of these listed availability `LOW`.

| Machine | $/h | Min vCPUs | $/vCPU-h | Raw steps/s | Compute $ per 1B steps |
|---|---|---|---|---|---|
| CPU pod, `cpu3c`, 32 vCPUs (no GPU) | 0.96 | 32 | 0.030 | 6,400–12,308 | 22–42 |
| RTX 4090, community | 0.34 | 12 | 0.028 | 2,400–4,615 | 20–39 |
| RTX 5090, community | 0.69 | 16 | 0.043 | 3,200–6,154 | 31–60 |
| A100 SXM 80 GB, secure | 1.79 | 32 | 0.056 | 6,400–12,308 | 40–78 |
| RTX A6000, secure | 0.59 | 9 | 0.066 | 1,800–3,462 | 47–91 |
| RTX 4090, secure | 0.89 | 12 | 0.074 | 2,400–4,615 | 54–103 |
| RTX 5090, secure | 1.19 | 16 | 0.074 | 3,200–6,154 | 54–103 |
| RTX PRO 6000, secure | 2.49 | 14 | 0.178 | 2,800–5,385 | 128–247 |
| H100 SXM, secure | 3.99 | 20 | 0.200 | 4,000–7,692 | 144–277 |

For this workload you're paying for CPU, so the premium GPUs are the worst value. The H100 SXM costs about 5× as much per step as the 32-vCPU A100 SXM's lower bound would suggest at equal CPU, and its extra GPU speed would sit idle with a small model. Among these, the A100 SXM has the best CPU allocation of the single-GPU options. The catalog doesn't say whether vCPUs scale with GPU count, and the 32-vCPU cap on CPU pods means going further needs several machines or a provider with larger CPU instances.

### Learning doesn't scale with throughput

More cores only buy steps. How those steps turn into progress depends on how you spend them. Holding k=8 and growing N forces a choice:

| vCPUs | N | T=2,048: batch | T=2,048: rollout RAM | Fixed 65,536 batch: T | Env RAM |
|---|---|---|---|---|---|
| 32 | 256 | 524k | 9 GB | 256 | 18 GB |
| 64 | 512 | 1.05M | 18 GB | 128 | 36 GB |
| 128 | 1,024 | 2.1M | 36 GB | 64 | 73 GB |
| 256 | 2,048 | 4.2M | 72 GB | 32 | 145 GB |
| 512 | 4,096 | 8.4M | 145 GB | 16 | 291 GB |

- **Keep T fixed:** batches grow, but updates per hour don't. Each update still needs 2,048 steps per env, about 49–88 s of collection at k=8 (11–16 s at k=1), however many cores you add.
- **Keep batch fixed:** updates get more frequent, but T shrinks. Pleines et al. found 2,048 worked better than 512 or 1,024 at their scale, so short rollouts are a real risk on this game. How this interacts with larger N is untested.
- **Per-env depth is the part that doesn't parallelize.** A 19,816-step episode takes about 0.9–1.7 minutes per env at k=1 and 7–13 minutes at k=8. No GPU or extra core shortens this. Over a 400M-step run, each env gets 12.5M steps at N=32, but only about 98k, or about 5 such episodes, at N=4,096.
- **Environments also overlap.** Envs starting from one save state repeat early-game play, so a larger batch adds less information than its size suggests. [Beukman et al. 2026](https://arxiv.org/abs/2603.06009) report that standard PPO configs plateau before about 10B interactions, and advise holding minibatch size and learning rate fixed while adding minibatches. Their results come from GPU-native simulators, not this game.

So parallel efficiency for learning will drop off before throughput does. I can't put a number on where without training runs, which this benchmark doesn't include.

### What to measure on a real machine

A short benchmark on the target machine would replace every assumption here:

- `b` per vCPU, and whether it degrades when all vCPUs are busy (the most important unknown)
- `g` for the update, and `L` and `c` for inference on CUDA
- `O(W)` at 32+ workers, plus an async comparison

## Recommendations

Roughly in order of measured gain.

1. **Run several envs per worker and batch inference on MPS.** Going from 12 × 1 to 12 × 8 was the largest gain in the tuned loop. Keep total envs within RAM.
2. **If you run about 12 envs in total, use CPU inference.** This gave +23%, updating on MPS and copying the weights back after each update.
3. **Keep the update on MPS.** It was about 15× faster per sample than CPU for the Nature CNN and more than 10× for IMPALA ×4.
4. **Turn sound off.** That saves 3–9% of emulator time for one flag.
5. **Replace the float observation path.** Integer operations would likely keep the averaging at far lower cost. The slice I tested gave about 2–12% more throughput, but it discards pixels and I didn't test its effect on learning, so check that before adopting it.
6. **Start with a small model.** Moving from Nature to IMPALA ×1 cut throughput by about 37–40% at 12 × 1. Move up only if the learning results justify it.
7. **For runs beyond about 1B steps, get more CPU cores before a bigger GPU.** Emulation is the bottleneck and scales with cores. As a rough estimate, not a measurement, 64 physical cores at 370–460 steps/s each gives about 24,000–29,000 raw steps/s before overhead. Synchronous overhead could cut that substantially, as covered in [Scaling on GPU machines](#scaling-on-gpu-machines).
8. **Don't shorten frame skip just for speed.** Step cost does scale with frames per step (293 µs at 4 frames, 4,544 µs at 48), but Pleines et al. chose 24 frames to give the game time to process each press, so a shorter skip may produce more wasted steps.

A more than 10× gain on one machine would need a faster emulator than PyBoy. I didn't evaluate any alternatives.

## Method and limitations

**Parallel Claude shells.** Other work was running on this machine during testing. The first benchmark pass was contaminated (the full loop read as low as 666 steps/s vs about 1,450 when clean), so I discarded those numbers except where noted.

The reruns wait until other processes use under about 1 core, and log how many cores those processes used during each measurement. Almost every reported run had under 1 core of background use. The exception is free-running N=4 in the scaling table (1.56 cores), so it may read slightly low. Rows in pass 1 with heavy background load, such as N=8 at 5.74 cores, aren't reported.

**Unexplained result.** Pass 1 free-running at N=4 and N=6 gave 1,197 and 1,688 steps/s against 1,850 and 2,613 in pass 2. Other processes logged under 0.5 cores in those pass 1 runs, so background load doesn't obviously explain it. The per-step CPU time was also higher in pass 1, so treat low-N free-running values as less certain.

**What this does not cover:**

- **Battles and later game states.** Benchmarks started from Red's room and Pallet Town. Battle cost is unknown, and game-state cost varied by a factor of 2 across what I did test.
- **Real training.** Policies were untrained and PPO targets were random. This is valid for timing but says nothing about learning. Policy behavior also changes the game states visited, and therefore emulator cost.
- **Richer environments.** The env reads 41 RAM addresses and tracks visited coordinates. Heavier reward logic, extra observations such as a visit map, or a real recurrent PPO loop with sequence batching would add cost.
- **Sustained load.** Full-loop windows were short, about 3–70 seconds. Thermal throttling over hours is not reflected.
- **Other hardware.** No CUDA GPU or many-core server was tested. MPS and Mac scheduler behavior won't transfer directly, so the GPU scaling section is a projection.
- **Unexplained inference gap.** MPS inference in the full loop took about 2.65 ms per batch of 12, versus 0.48 ms in isolation. Sampling, three separate `.cpu()` syncs and data transfer are the likely causes, but I didn't break this down.
- **Hyperparameters.** T, epochs and minibatch size are benchmark settings, not tuned values.

The benchmark scripts and raw logs are in the session scratchpad, which is temporary: a temporary session folder that no longer exists. The main log is `gate.log` there.

## Sources

- [Pleines et al. 2025, *Playing Pokémon Red via Deep Reinforcement Learning*](https://arxiv.org/abs/2502.19920)
- [`pokemonred_puffer` `config.yaml`](https://github.com/drubinstein/pokemonred_puffer)
- [`PokemonRedExperiments` `baseline_fast_v2.py`](https://github.com/PWhiddy/PokemonRedExperiments), which also uses 24 frames per action
- [PyBoy documentation](https://github.com/Baekalfen/PyBoy)
