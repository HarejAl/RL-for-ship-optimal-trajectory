"""
Pure-RL training of the wind-aware agent on the GPU (PLAN.md step B2-B3). No DP teacher.

Thousands of ships (`gpu_env.BatchShipEnv`) sail generated wind fields at once; each episode
draws a field, a windage ratio kappa, a wind strength, a start and a goal. The field bank is
refreshed during training, so the agent never sees the same weather twice for long, and never
the held-out benchmark seeds. PPO with GAE; the reward is the exact DP stage cost plus
potential-based shaping (see gpu_env.py), so the DP optimum is still the right yardstick.

    python train_ppo.py --tag ppo_v1 --hours 12
    python train_ppo.py --tag smoke --iters 20 --n-envs 512        # quick check

Outputs: models/<tag>.pt (+ .json obs config; best on validation), models/<tag>_last.pt,
output/logs/<tag>/log.jsonl (one line per iteration). Evaluate with
`python benchmark_dp.py --n-cases 30 --model models/<tag>.pt --kappa 0.5`.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from gpu_env import FieldBank, BatchShipEnv
from ppo_policy import ActorCritic, save_checkpoint
from wind_obs import EVAL_SEED_BASE

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="ppo_v1")
    ap.add_argument("--n-envs", type=int, default=2048)
    ap.add_argument("--n-steps", type=int, default=64, help="rollout length per iteration")
    ap.add_argument("--iters", type=int, default=None, help="stop after this many iterations")
    ap.add_argument("--hours", type=float, default=None, help="stop after this wall-clock time")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--n-fields", type=int, default=4096, help="fields resident in the GPU bank")
    ap.add_argument("--refresh-every", type=int, default=10, help="iterations between bank refreshes")
    ap.add_argument("--refresh-n", type=int, default=128, help="fields regenerated per refresh")
    ap.add_argument("--kappa", type=float, nargs=2, default=[0.05, 0.6])
    ap.add_argument("--wind-mult", type=float, nargs=2, default=[0.25, 1.0])
    ap.add_argument("--radius-start", type=float, default=1.0, help="goal-radius curriculum start")
    ap.add_argument("--radius-final", type=float, default=0.5)
    ap.add_argument("--radius-success", type=float, default=0.7, help="shrink when success rate exceeds")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def make_eval(args, device):
    """Fixed validation cases on held-out fields (EVAL_SEED_BASE, never trained on)."""
    rng = np.random.default_rng(12345)
    n = args.eval_n
    n_fields = 64
    bank = FieldBank(n_fields, device, seed_start=EVAL_SEED_BASE)
    env = BatchShipEnv(n, bank, max_steps=600, gamma=args.gamma, seed=1)
    starts, goals = [], []
    while len(starts) < n:
        s, g = rng.uniform(0, 10, 2), rng.uniform(0, 10, 2)
        if np.linalg.norm(s - g) >= 4.0:
            starts.append(s)
            goals.append(g)
    case = dict(fidx=np.arange(n) % n_fields, start=np.array(starts), goal=np.array(goals),
                kappa=np.tile([0.1, 0.3, 0.5, 0.6], n // 4 + 1)[:n],
                mult=np.tile([1.0, 1.0, 0.6, 0.3], n // 4 + 1)[:n])
    return env, case


@torch.no_grad()
def evaluate(model, env, case):
    env.radius = env.goal_radius
    env.set_cases(**case)
    alive = torch.ones(env.n, dtype=torch.bool, device=env.dev)
    success = torch.zeros_like(alive)
    J = torch.zeros(env.n, device=env.dev)
    for _ in range(env.max_steps):
        a = model.mean(env.obs())
        _, _, term, trunc, info = env.step(a, auto_reset=False)
        fin = alive & info["done"]
        success |= fin & info["success"]
        J = torch.where(fin, info["J"], J)
        alive &= ~info["done"]
        # finished ships are frozen in place: zero their velocity so they cannot re-trigger
        env.vel[~alive] = 0.0
        if not alive.any():
            break
    k = torch.as_tensor(case["kappa"], device=env.dev)
    out = dict(eval_success=success.float().mean().item(),
               eval_J=J[success].mean().item() if success.any() else float("nan"))
    for kv in sorted(set(case["kappa"].tolist())):
        m = k == kv
        out[f"eval_success_k{kv:g}"] = success[m].float().mean().item()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(args.device)
    models_dir = os.path.join(SCRIPT_DIR, "models")
    log_dir = os.path.join(SCRIPT_DIR, "output", "logs", args.tag)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    t0 = time.time()
    bank = FieldBank(args.n_fields, dev)
    env = BatchShipEnv(args.n_envs, bank, kappa_range=args.kappa, wind_mult_range=args.wind_mult,
                       gamma=args.gamma, seed=args.seed)
    env.radius = args.radius_start
    eval_env, eval_case = make_eval(args, dev)
    print(f"field bank of {bank.n} built in {time.time() - t0:.1f}s on {dev}", flush=True)

    model = ActorCritic().to(dev)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=dev, weights_only=False)["state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    N, T = args.n_envs, args.n_steps
    obs = env.obs()
    # maps are stored in half precision (they are O(1) and smooth): halves the GPU memory, which
    # matters on a card shared with other jobs (WDDM spills to system RAM when it is full)
    buf_obs = {k: torch.zeros((T, N) + v.shape[1:], device=dev,
                              dtype=torch.float32 if k == "vec" else torch.float16) for k, v in obs.items()}
    buf_act = torch.zeros(T, N, 2, device=dev)
    buf_logp = torch.zeros(T, N, device=dev)
    buf_val = torch.zeros(T, N, device=dev)
    buf_rew = torch.zeros(T, N, device=dev)
    buf_done = torch.zeros(T, N, device=dev)
    best = (-1.0, -float("inf"))
    it, steps, t_start = 0, 0, time.time()
    ep_window = dict(success=[], oob=[], J=[], ret=[])
    ep_ret = torch.zeros(N, device=dev)
    log_f = open(os.path.join(log_dir, "log.jsonl"), "a")

    while True:
        it += 1
        if args.iters and it > args.iters:
            break
        if args.hours and time.time() - t_start > args.hours * 3600:
            break
        if args.hours or args.iters:      # linear learning-rate decay over the budget
            frac = (time.time() - t_start) / (args.hours * 3600) if args.hours else (it - 1) / args.iters
            for g in opt.param_groups:
                g["lr"] = args.lr * max(1.0 - frac, 0.05)

        # ---------------------------------------------------------------- rollout
        t_roll = time.time()
        n_succ = n_done = 0
        with torch.no_grad():
            for t in range(T):
                for k in obs:
                    buf_obs[k][t] = obs[k]
                    obs[k] = buf_obs[k][t].float()   # act on exactly what the update will see
                dist = model.dist(obs)
                a = dist.sample()
                buf_act[t] = a
                buf_logp[t] = dist.log_prob(a).sum(-1)
                buf_val[t] = model.value(obs)
                obs, r, term, trunc, info = env.step(a)
                if trunc.any():   # bootstrap time-limit truncations with V(final state)
                    r = r + args.gamma * trunc.float() * model.value(info["final_obs"])
                buf_rew[t] = r
                buf_done[t] = info["done"].float()
                ep_ret += r
                done = info["done"]
                if done.any():
                    ep_window["success"].append(info["success"][done].float().cpu())
                    ep_window["oob"].append(info["oob"][done].float().cpu())
                    ep_window["J"].append(info["J"][done & info["success"]].cpu())
                    ep_window["ret"].append(ep_ret[done].cpu())
                    n_succ += int(info["success"][done].sum())
                    n_done += int(done.sum())
                    ep_ret[done] = 0.0
            last_val = model.value(obs)
        steps += N * T
        t_roll = time.time() - t_roll

        # goal-radius curriculum on the success rate of this iteration's episodes
        if n_done and n_succ / n_done > args.radius_success and env.radius > args.radius_final:
            env.radius = max(args.radius_final, env.radius * 0.95)

        # -------------------------------------------------------------------- GAE
        adv = torch.zeros(T, N, device=dev)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * nv * nonterm - buf_val[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_val

        # ----------------------------------------------------------------- update
        t_upd = time.time()
        flat = {k: v.reshape((T * N,) + v.shape[2:]) for k, v in buf_obs.items()}
        f_act, f_logp = buf_act.reshape(-1, 2), buf_logp.reshape(-1)
        f_adv, f_ret = adv.reshape(-1), ret.reshape(-1)
        stats = dict(pg=0.0, vf=0.0, kl=0.0, clipfrac=0.0)
        n_mb = 0
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=dev)
            for s in range(0, T * N, args.minibatch):
                mb = perm[s:s + args.minibatch]
                o = {k: v[mb].float() for k, v in flat.items()}
                d = model.dist(o)
                logp = d.log_prob(f_act[mb]).sum(-1)
                ratio = (logp - f_logp[mb]).exp()
                a_mb = f_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb).mean()
                vf = 0.5 * (model.value(o) - f_ret[mb]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + args.vf * vf - args.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                opt.step()
                with torch.no_grad():
                    stats["pg"] += pg.item()
                    stats["vf"] += vf.item()
                    stats["kl"] += ((ratio - 1) - (logp - f_logp[mb])).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                n_mb += 1
        stats = {k: v / n_mb for k, v in stats.items()}
        t_upd = time.time() - t_upd

        if it % args.refresh_every == 0:
            bank.replace(np.random.default_rng(it).choice(bank.n, args.refresh_n, replace=False))

        # -------------------------------------------------------------------- log
        cat = lambda key: torch.cat(ep_window[key]) if ep_window[key] else torch.zeros(0)
        row = dict(iter=it, steps=steps, hours=(time.time() - t_start) / 3600,
                   sps=int(N * T / (t_roll + t_upd)), t_roll=round(t_roll, 2), t_upd=round(t_upd, 2),
                   lr=opt.param_groups[0]["lr"], radius=env.radius, std=model.log_std.exp().mean().item(),
                   episodes=int(cat("success").numel()), success=cat("success").mean().item(),
                   oob=cat("oob").mean().item(), J=cat("J").mean().item() if cat("J").numel() else None,
                   ret=cat("ret").mean().item(), fields_seen=int(bank.next_seed - bank.seeds.min()), **stats)
        ep_window = dict(success=[], oob=[], J=[], ret=[])

        if it % args.eval_every == 0 or it == 1:
            model.eval()
            row.update(evaluate(model, eval_env, eval_case))
            model.train()
            score = (row["eval_success"], -row["eval_J"] if row["eval_J"] == row["eval_J"] else -1e9)
            if score > best:
                best = score
                save_checkpoint(os.path.join(models_dir, f"{args.tag}.pt"), model, env.cfg,
                                extra=dict(iter=it, steps=steps, **{k: v for k, v in row.items()
                                                                    if k.startswith("eval_")}))
                row["saved_best"] = True
        if it % args.save_every == 0:
            save_checkpoint(os.path.join(models_dir, f"{args.tag}_last.pt"), model, env.cfg,
                            extra=dict(iter=it, steps=steps))
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        msg = (f"it {it:5d}  {steps / 1e6:8.1f}M steps  {row['sps']:6d} sps  succ {row['success']:.2f}  "
               f"oob {row['oob']:.2f}  J {row['J'] if row['J'] is None else round(row['J'], 2)}  "
               f"r={env.radius:.2f}  std {row['std']:.2f}  kl {stats['kl']:.4f}")
        if "eval_success" in row:
            msg += f"  | EVAL succ {row['eval_success']:.2f} J {row['eval_J']:.2f}"
        print(msg, flush=True)

    save_checkpoint(os.path.join(models_dir, f"{args.tag}_last.pt"), model, env.cfg, extra=dict(iter=it, steps=steps))
    print(f"done: {steps / 1e6:.1f}M steps in {(time.time() - t_start) / 3600:.2f} h; best eval {best}")


if __name__ == "__main__":
    main()
