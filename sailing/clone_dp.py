"""
DP-taught sailing agent: behaviour cloning + DAgger on random wind fields.

The student sees exactly what the RL agent sees (goal-aligned wind map + 10-vector, see
rl_env.py) and picks one of `N_BINS` headings relative to the goal bearing. Classification
rather than regression on purpose: near a tack decision the teacher's label flips between
two headings ~90 deg apart, and a regression net would average them into the no-go zone.

    round 0 : solve DP for random (field, start, goal) cases; label the DP route and random
              states (position, heading) scattered around it
    round k : sail the current student, label every state it visits with the DP action,
              aggregate, retrain                                       (DAgger)

    python sailing/clone_dp.py --tag clone_fields --cases0 400 --rounds 3
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, REPO_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sailing.rl_env import SailRLEnv, FieldPoolScenarios, RL_PARAMS
from sailing.rl_eval import evaluate, dp_reference, summary, run_episode
from sailing.dp_sail import SailDP

MODEL_DIR = os.path.join(REPO_DIR, "models", "sail")
LOG_DIR = os.path.join(REPO_DIR, "output", "sailing", "rl")
N_BINS = 72
MAP_RES = 16


def a_to_bin(a):
    return np.clip(np.round((np.asarray(a) + 1.0) / 2.0 * N_BINS).astype(int) % N_BINS, 0, N_BINS - 1)


def bin_to_a(k):
    a = np.asarray(k) * 2.0 / N_BINS - 1.0
    return np.where(a >= 1.0, a - 2.0, a)


class Student(nn.Module):
    def __init__(self, map_res=MAP_RES, n_bins=N_BINS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        n_flat = 64 * ((map_res + 3) // 4) ** 2
        self.map_fc = nn.Sequential(nn.Linear(n_flat, 128), nn.ReLU())
        self.vec_fc = nn.Sequential(nn.Linear(10, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, n_bins))

    def forward(self, vec, m):
        return self.head(torch.cat((self.map_fc(self.cnn(m)), self.vec_fc(vec)), dim=1))


def student_policy(net, device):
    @torch.no_grad()
    def pol(env):
        o = env.observe()
        logits = net(torch.as_tensor(o["vec"][None], device=device), torch.as_tensor(o["map"][None], device=device))
        return env.heading_for(float(bin_to_a(int(logits.argmax()))))
    return pol


# ------------------------------------------------------------------ labelling
def set_state(env, x, y, h):
    e = env.sail
    e.state = np.array([x, y, h], dtype=float)
    e.pending, e._started, e.steps = 0.0, True, 1


def label(env, dp, x, y, h):
    """(obs, bin) for a boat at (x, y) on heading h with no manoeuvre pending."""
    set_state(env, x, y, h)
    o = env.observe()
    b = np.arctan2(env.sail.goal[1] - y, env.sail.goal[0] - x)
    hd = dp.act(x, y, h)
    a = np.arctan2(np.sin(hd - b), np.cos(hd - b)) / np.pi
    return o, int(a_to_bin(a))


class Buffer:
    def __init__(self):
        self.vec, self.map, self.y = [], [], []

    def add(self, o, k):
        self.vec.append(o["vec"]); self.map.append(o["map"]); self.y.append(k)

    def __len__(self):
        return len(self.y)

    def arrays(self):
        return np.stack(self.vec), np.stack(self.map), np.array(self.y)


def solve_case(env, case):
    wind, start, goal = case
    dp = SailDP(wind, env.sail.polar, RL_PARAMS, goal, nx=81, ny=81)
    dp.solve()
    return dp


def collect_expert(env, rng, case, buf, n_random=60):
    wind, start, goal = case
    dp = solve_case(env, case)
    pol = dp.policy()
    r = run_episode(env, lambda e: pol(e.sail), case)
    traj = r["traj"]
    env.reset(options=dict(wind=wind, start=start, goal=goal))
    for x, y, h in traj[:-1]:
        buf.add(*label(env, dp, x, y, h))
    # random states around the route: position jitter + random heading
    for _ in range(n_random):
        x, y, _ = traj[rng.integers(len(traj))]
        x = float(np.clip(x + rng.normal(0, 1.0), -0.8, 10.8))
        y = float(np.clip(y + rng.normal(0, 1.0), -0.8, 10.8))
        if np.hypot(x - goal[0], y - goal[1]) <= RL_PARAMS.goal_radius:
            continue
        buf.add(*label(env, dp, x, y, rng.uniform(-np.pi, np.pi)))
    return r


def collect_dagger(env, case, buf, net, device, max_steps=600):
    wind, start, goal = case
    dp = solve_case(env, case)
    pol = student_policy(net, device)
    env.reset(options=dict(wind=wind, start=start, goal=goal))
    visited = []
    info = env.sail._info()
    for _ in range(max_steps):
        if env.sail.pending <= 1e-12:
            visited.append(env.sail.state.copy())
        _, _, term, trunc, info = env.sail.step(np.array([pol(env)]))
        if term or trunc:
            break
    env.reset(options=dict(wind=wind, start=start, goal=goal))
    # cap per-episode states so looping failures do not flood the buffer
    idx = np.linspace(0, len(visited) - 1, min(len(visited), 150)).astype(int) if visited else []
    for i in idx:
        x, y, h = visited[i]
        buf.add(*label(env, dp, float(x), float(y), float(h)))
    return info["success"]


# ------------------------------------------------------------------ training
def train(net, buf, device, epochs, lr=1e-3, batch=512, log=print):
    V, M, Y = buf.arrays()
    V, M, Y = (torch.as_tensor(V, device=device), torch.as_tensor(M, device=device),
               torch.as_tensor(Y, device=device))
    n = len(Y)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((n + batch - 1) // batch))
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, acc = 0.0, 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            logits = net(V[j], M[j])
            # soft-ish target: neighbouring bins (+-5 deg) are nearly as good
            tgt = torch.zeros_like(logits)
            tgt[torch.arange(len(j)), Y[j]] = 0.8
            tgt[torch.arange(len(j)), (Y[j] + 1) % N_BINS] = 0.1
            tgt[torch.arange(len(j)), (Y[j] - 1) % N_BINS] = 0.1
            loss = -(tgt * F.log_softmax(logits, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(j)
            err = (logits.argmax(1) - Y[j]) % N_BINS
            acc += float(((err <= 1) | (err == N_BINS - 1)).sum())
        if ep == epochs - 1 or ep % 5 == 0:
            log(f"    epoch {ep:>3}  loss {tot / n:.3f}  acc(+-5deg) {acc / n:.1%}")
    net.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="clone_fields")
    ap.add_argument("--cases0", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cases-per-round", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{args.tag}_log.jsonl")
    open(log_path, "w").close()

    def log(msg):
        print(msg, flush=True)

    rng = np.random.default_rng(args.seed)
    scen = FieldPoolScenarios(500)
    env = SailRLEnv(scen, map_res=MAP_RES)
    ref = dp_reference("fields", args.eval_n)
    log(f"DP on held-out fields: success {ref['success'].mean():.0%}, mean {ref['t'].mean():.1f} h, "
        f"tacks {ref['tacks'].mean():.1f}, solve {ref['solve_s'].mean():.2f} s")

    buf = Buffer()
    net = Student().to(device)
    t0 = time.time()
    for rnd in range(args.rounds + 1):
        n_cases = args.cases0 if rnd == 0 else args.cases_per_round
        succ = []
        for i in range(n_cases):
            case = scen(rng)
            try:
                if rnd == 0:
                    collect_expert(env, rng, case, buf)
                else:
                    succ.append(collect_dagger(env, case, buf, net, device))
            except ValueError:
                continue
        log(f"round {rnd}: {len(buf):,} labels ({time.time() - t0:.0f} s)"
            + (f", student train success while collecting {np.mean(succ):.0%}" if succ else ""))
        train(net, buf, device, args.epochs, log=log)
        res = evaluate(student_policy(net, device), "fields", args.eval_n, env=SailRLEnv(None, map_res=MAP_RES))
        s = summary(res, ref)
        s.update(round=rnd, labels=len(buf), wall_min=(time.time() - t0) / 60)
        log(f"  held-out: success {s['success']:.0%} (DP {s['dp_success']:.0%})  median gap "
            f"{s['median_gap_pct']:+.1f}%  tacks {s['tacks']:.1f} (DP {s['dp_tacks']:.1f})  "
            f"{s['ms_per_episode']:.0f} ms/episode")
        with open(log_path, "a") as f:
            f.write(json.dumps(s) + "\n")
        torch.save(net.state_dict(), os.path.join(MODEL_DIR, f"{args.tag}_r{rnd}.pt"))
    log("done")


if __name__ == "__main__":
    main()
