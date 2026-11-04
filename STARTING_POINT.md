# Where to start teaching an agent to play Pokemon Red

> Some claims here were later corrected; see `docs/CONFLICT_RESOLUTIONS.md`.

**Recommendation:** write a small Gymnasium environment on PyBoy 2.9 that copies the published baseline from Pleines et al. (2025), train it with Stable-Baselines3 PPO on MPS, and make "beat Brock" the first goal.

That baseline is the best starting point because it is the simplest setup with published results to compare against: 5 seeds, milestone rates and reward ablations. It beat Brock at a 0.99 rate, against a human reference of 1.00.

Hugging Face is not the trainer. It has no PPO trainer that fits this problem, though it is useful for sharing checkpoints later.

This is not a plan to finish the game. The baseline covers about 20% of it and never gets past Cerulean. Going further takes the extra machinery described in [After Brock](#after-brock).

## Options considered

| Option | Verdict | Why |
|---|---|---|
| Hugging Face TRL trainers (DPO, GRPO and others) | No | They train language models on text. There is no environment loop for an emulator. |
| Hugging Face Deep RL Course, Unit 8 | Read only | A good PPO walkthrough, derived from CleanRL. Its defaults (4 envs, 128 steps) target LunarLander and CartPole, and its install notes use old `gym`. |
| Pokemon Red models on the Hub | No | Nearly all are language-model fine-tunes. The one PPO-related repo found, `aboundy/pokemonRed`, has 0 downloads. |
| Whidden `PokemonRedExperiments` v2 | Reference | The original project. It pins PyBoy 2.4.0, SB3 2.3.2 and torch 2.5.0, and uses 64 envs. Its reward differs from the paper's, so the paper's results don't apply to it. |
| **Pleines et al. baseline, rebuilt** | **Start here** | Simple, with published milestone rates and ablations to check against. |
| PokeRL (2026) | Borrow ideas | Early-game tasks only, with a 50% win rate against the rival. Its anti-loop measures are worth trying if looping appears. |
| PufferLib / Rubinstein et al. | Later | The furthest result: it "beats Pokémon with a few caveats". It relies on 25+ reward parameters, scripted behaviors and state-sharing across envs, which would hide bugs in a first build. |

## The first configuration

### Start state

Start after Oak's Parcel, with Squirtle, fast text and battle animations off. This is the paper's Fast setup, and the table shows why:

| Experiment | Milestones (of 7) | Beat Brock | Reach Cerulean | Steps to Brock |
|---|---|---|---|---|
| Human | 7.00 ± 0.0 | 1.00 | 1.00 | 5403 ± 3824 |
| **Fast (Squirtle)** | **4.40 ± 0.8** | **0.98** | **0.93** | **3753 ± 990** |
| Baseline (Squirtle) | 4.19 ± 0.2 | 0.99 | 0.85 | 5587 ± 1235 |
| GRU | 3.93 ± 2.2 | 0.76 | 0.49 | 5624 ± 780 |
| Choose Starter | 3.43 ± 1.7 | 0.79 | 0.75 | 6485 ± 578 |
| Charmander | 3.03 ± 1.6 | 0.69 | 0.53 | 7029 ± 1132 |
| Bulbasaur | 2.89 ± 0.1 | 0.95 | 0.00 | 9929 ± 841 |

Source: Pleines et al., Table II.

No state file for this exists yet. `init.state` in Whidden's repo targets an older PyBoy, and I didn't test whether it loads on 2.9. The simplest route is to play to that point once and call `save_state`.

### Actions

Use seven buttons: A, B, Start, Up, Down, Left, Right. Each action holds the button for 8 frames, inside a 24-frame decision. I measured this on your ROM:

| Hold / period (frames) | Result |
|---|---|
| 1 / 24 | Doesn't register: 0 tiles in 4 presses |
| 2 / 24 | Turns in place on a change of direction, then walks |
| 4, 8, 12 or 16 / 24 | The first press moves one tile, until the player reaches a wall |
| 8 / 16 | The first press is lost |
| 8 / 32 | Works, but uses 8 more frames per decision |

**Don't add double-press handling.** PokeRL says a press only turns the player and a second one moves. At 8/24, one press both turned and moved, including on a 180° turn. Its behavior matches what I saw with very short holds, so the likely cause is its press timing, though I didn't read its code.

Keep Start even though PokeRL removes it, because items and field moves need the menu later. If Start spam becomes a problem, deal with that when it appears.

### Observations

Follow the paper:

- **Screen:** the current and two previous grayscale frames, downsampled to 72×80.
- **Visited map:** a 48×48 binary crop centered on the player.
- **Game state:** each party member's HP and level, plus the event flags.

`RAM_MAP_REPORT.md` suggests richer battle inputs, such as moves, types and stat stages. The paper deliberately leaves out species, stats and movesets and still beats Brock, so I would add those only if battles turn out to be what's blocking progress. `ram_map.py` makes that easy later.

### Reward

These are the four terms from the paper, §II-F:

| Term | Value |
|---|---|
| Event | +2 per completed event |
| Navigation | +0.005 per new overworld coordinate in the episode |
| Heal | 2.5 × Σ (HP gained / max HP) over the party |
| Level | 0.5 × min(Σlvl, (Σlvl − 22)/4 + 22) |

The navigation weight is the sensitive one. Removing it dropped the agent to 0 milestones, and multiplying it by 10 dropped Beat Brock from 0.99 to 0.04.

Guard against these exploits and errors from the start:

- **Blackouts pay out as healing.** A blackout fully restores HP (`RAM_MAP_REPORT.md`, trap 7), so don't pay heal reward for it. Whidden's code treats HP going from 0 to nonzero as a death.
- **Healing can be farmed.** Bulbasaur runs averaged 399 heals and never reached Cerulean. Log heals per episode.
- **Pay the level term as a change between steps.** The paper writes it as a total. Paying the total every step would be a large bug. Whidden pays changes; confirm against the paper's code.
- **Subtract the event count at reset.** Flags already set in the start state shouldn't earn reward. The range length is also unsettled: `ram_map.py` uses 320 bytes (recalled, end unconfirmed) and Whidden uses 311. Pick one and record it.

### Episodes

Start with a budget of 10,240 steps and add 2,048 per completed event. Time is the only ending condition, so report it as `truncated`, never `terminated`. SB3 2.9.2 then bootstraps the value correctly (confirmed in its source).

The growing budget has a purpose. Envs fall out of sync, so they don't all reset together, which the paper uses to reduce catastrophic forgetting.

### PPO settings

These are from the paper's Table I, written for SB3:

```python
PPO(
    "MultiInputPolicy", env,
    device="mps",            # "auto" falls back to CPU on a Mac
    n_steps=2048,            # with 32 envs: 65,536 per batch
    batch_size=8192,         # 8 minibatches
    n_epochs=3,
    learning_rate=3e-4,
    gamma=0.997, gae_lambda=0.95,
    clip_range=0.2, clip_range_vf=0.2,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(optimizer_class=torch.optim.AdamW, activation_fn=torch.nn.ReLU),
)
```

Three things will differ from the paper unless handled:

- **Device:** SB3's `auto` only looks for CUDA, so MPS has to be requested explicitly.
- **Optimizer and value clipping:** SB3 defaults to Adam and no value clipping, while the paper uses AdamW and clips. The settings above fix both.
- **Network size:** my test network had 1.10M parameters against the paper's roughly 2M, because the architectures differ. Adjust this first if results fall short.

Skip the GRU for now. It helped with Bill's quest (0.48), but it was less consistent overall and took 24 days per run on the authors' CPUs.

## What this costs on your Mac

I measured this on the M4 Pro (14 cores, 24 GB), using PyBoy 2.9.0, SB3 2.9.2, torch 2.15.0 and Python 3.12.

These are ballparks. Other Claude shells were using the CPU during the tests, so a quiet machine should do somewhat better.

| Measurement | Ballpark |
|---|---|
| One env | about 450 steps/s |
| All cores, envs only | about 4,000–5,000 steps/s total |
| Memory per env | about 100 MB |
| Reset | a few ms |
| **Full PPO, 10 envs, MPS** | **about 1,000 steps/s, or about 4M steps/h** |
| Full PPO, 10 envs, CPU (4 threads) | about 300 steps/s, or about 1M steps/h |

Two conclusions:

- **Use MPS.** PPO updates took seconds on MPS and about a minute on CPU. The CPU run was limited to 4 threads, so the gap may be smaller with more, but it is still the clear default.
- **Look at the rollout overhead before buying more compute.** Rollouts reached only about a third of what the envs can do alone. I didn't profile the gap. Passing observations between processes and running inference are the likely causes.

**Rough time per seed, at about 4M steps/h:**

| Goal | Steps (read by eye from the paper's Fig. 3) | Time |
|---|---|---|
| Past Brock | about 100M | about a day |
| Cerulean | about 200M | about 2 days |
| Full 400M run | 400M | about 4 days |

Five seeds to Brock would take most of a week in sequence. This uses the 10-env rate; I didn't measure PPO at 32 envs. Memory isn't the limit: 32 envs plus the rollout buffer need about 5 GB.

## Build order

Each stage should pass before the next one starts.

1. **Env correctness, no learning.** Test that one press moves one tile, that each reward term fires once and only once, that a blackout pays no heal, and that truncation works.
2. **Random baseline from the real start state.** My 20-seed test ran from the bedroom, not the post-parcel state. Repeat it from your real start.
3. **Single seed to 20M steps, on MPS.** Check that unique coordinates clearly beat random. Don't expect milestones yet: the paper's Mt. Moon curves stay near zero until about 20M. Log PPO's KL, clip fraction and explained variance, the reward from each term, and action frequencies. Watch recorded episodes too, because reward curves can hide looping.
4. **Single seed to 100M.** Pass if Beat Brock is well above 0.5 and rising toward the paper's 0.98.
5. **Five seeds.** Compare against Table II. Report milestone rates and steps to milestone, never total return, since return changes whenever the reward does.

**Evaluate with a sampled policy, not the highest-probability action.** The emulator is deterministic: replaying 3,000 actions twice gave identical memory. A deterministic policy would just repeat one run, so 30 evaluation episodes would count as one.

**House exit is a weak test here.** A random policy reached Pallet Town in 11 of 20 runs within 2,048 decisions. PokeRL's house-exit task (65% after 150k steps) therefore shows little at that budget. Its step limit isn't stated, so the two aren't directly comparable.

## After Brock

These are the next things to try if they become necessary, roughly in order:

1. **Save-state sharing.** PyBoy's save and load are cheap, as measured above. Rubinstein et al. load every env from the agent that completes a required objective, and report that the policy doesn't improve without this. The cost is less generalization.
2. **A GRU**, for multi-step quests such as Bill's.
3. **Better exploration signals**, such as RND or curiosity, since the paper shows how fragile the navigation weight is.
4. **Rewards and scripts for HMs.** The Cut tree past Vermilion stopped the paper's agents, and Rubinstein et al. needed scripts and many game-specific rewards. Expect this to be the hard part.

An agent that learns the whole game by itself from pixels and simple rewards is still an open problem. Keep that in mind when setting goals past Cerulean.

## What I didn't verify

- **No training was run.** This recommendation rests on the paper and on my throughput tests, and nothing here shows that my SB3 rebuild reproduces the paper. Stages 3 and 4 test that.
- **The paper's code wasn't read.** Read the [neroRL `poke_red` branch](https://github.com/MarcoMeter/neroRL/tree/poke_red) before building. It should settle how the level reward and event flags are computed.
- **Movement was tested in one bedroom only.** Stage 1 should repeat the test outdoors.
- **Rubinstein's training scale is unknown.** The pages I read give no steps, hardware or time for its results, so I can't estimate the cost of a full-game run.
- **The benchmark env is partial.** It builds full observations but computes only the navigation and event rewards, so a real env will be a little slower.
- **The Fig. 3 steps are approximate.** They were read by eye from a plot without error bands.

## Sources

- Pleines et al., [Pokemon Red via Reinforcement Learning](https://arxiv.org/abs/2502.19920v2) (2025), read from the PDF: hyperparameters, rewards, Tables II and III, Fig. 3
- Mudireddy & Patibandla, [PokeRL](https://arxiv.org/abs/2604.10812) (2026)
- Rubinstein et al., [PokeRL docs](https://drubinstein.github.io/pokerl/): [results](https://drubinstein.github.io/pokerl/docs/chapter-4/results/), [swarm](https://drubinstein.github.io/pokerl/docs/chapter-3/swarm/), [rewards](https://drubinstein.github.io/pokerl/docs/chapter-2/rewards/), [environment](https://drubinstein.github.io/pokerl/docs/chapter-2/env-setup/). These were read through a summarizing fetch, not line by line.
- Whidden, [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments): [v2 env](https://raw.githubusercontent.com/PWhiddy/PokemonRedExperiments/master/v2/red_gym_env_v2.py), [v2 training script](https://raw.githubusercontent.com/PWhiddy/PokemonRedExperiments/master/v2/baseline_fast_v2.py)
- [PyBoy docs](https://docs.pyboy.dk/) and [README](https://github.com/Baekalfen/PyBoy)
- [Hugging Face Deep RL Course, Unit 8](https://huggingface.co/learn/deep-rl-course/unit8/hands-on-cleanrl), [TRL docs](https://huggingface.co/docs/trl), [`aboundy/pokemonRed`](https://huggingface.co/aboundy/pokemonRed)
- [pret/pokered `roms.sha1`](https://raw.githubusercontent.com/pret/pokered/master/roms.sha1): your ROM's SHA-1 matches the `pokered` entry
- `RAM_MAP_REPORT.md` and `ram_map.py` in this directory, written by another session
- SB3 2.9.2 and PyBoy 2.9.0, from installed source and direct measurement
