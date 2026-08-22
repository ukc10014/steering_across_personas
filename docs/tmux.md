# Surviving a dropped connection on the pod

## The problem this solves, and the one it doesn't

SSH into the pod, start `claude`, lose your internet: the SSH session ends, its processes get
SIGHUP, and Claude Code dies with the conversation in it. tmux fixes exactly that — the
session keeps running on the pod and you reattach to the same process.

**tmux does not survive a pod stop or a RunPod migration.** Those destroy the container and
everything in it, tmux included. The 2026-08-20 loss was a migration, not a disconnect, so
tmux would not have helped. What protects against *that* is what the pipeline already does:

- long jobs launched with `setsid nohup`, so they own their process group
- every extraction resumable (`2c_caa_activations.py` skips complete cells and re-does
  truncated ones)
- all state on `/workspace`, the network volume, which is a separate resource from the pod

Use both. They cover different failures.

## Setup

Already done — `bootstrap.sh` installs tmux if the container lacks it (idempotent, so it
self-heals after a migration) and points `TMUX_CONF` at `/workspace/.tmux.conf`, which lives
on the volume.

## Use

```bash
source /workspace/bootstrap.sh
tm                      # attach to session "main", creating it if absent
```

`tm` is `tmux -f "$TMUX_CONF" new-session -A -s main`. The `-A` is what makes it one command
for both cases — no need to remember whether the session already exists.

Then start `claude` inside it. After a dropped connection:

```bash
ssh <pod>
source /workspace/bootstrap.sh && tm      # back in the same session
```

## Worth knowing

| keys | does |
|---|---|
| `Ctrl-b d` | detach (session keeps running) |
| `Ctrl-b c` | new window |
| `Ctrl-b n` / `p` | next / previous window |
| `Ctrl-b [` | scroll mode; `q` exits (mouse scroll also works) |
| `Ctrl-b ?` | list every binding |

`tmux ls` shows live sessions; `tmux kill-session -t main` ends one.

Scrollback is set to 100k lines, which matters because an 80-minute extraction prints a lot
and the interesting part is usually the beginning.

## The pattern that has actually held up here

For anything long, don't rely on the terminal at all — tmux included:

```bash
setsid nohup bash scripts/run_random_ladder.sh > /workspace/run.log 2>&1 </dev/null &
```

Then tail the log from wherever. `setsid` gives the job its own session, so it is immune to
both the SSH drop and to tmux itself being killed. tmux is for keeping *interactive* work
(the Claude Code conversation) alive; `setsid` is for keeping *jobs* alive. Yesterday's
ladder survived a Claude Code restart for exactly this reason.
