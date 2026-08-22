#!/bin/bash
# Pod environment bootstrap. Sourced from ~/.bashrc on every shell.
#
# TWO COPIES, KEPT BYTE-IDENTICAL:
#   /workspace/bootstrap.sh                  <- the LIVE one; ~/.bashrc sources this path
#   <repo>/scripts/bootstrap.sh              <- the tracked mirror, so it survives a volume rebuild
#
# The live copy is a real file, deliberately NOT a symlink into the repo: on a fresh volume
# the repo may not be cloned yet, and bootstrap is what sets up the environment you clone
# with -- a dangling symlink there would break every shell with no way back.
#
# After editing either one, sync and verify:
#   cp /workspace/bootstrap.sh <repo>/scripts/bootstrap.sh   # live -> repo (usual direction)
#   cp <repo>/scripts/bootstrap.sh /workspace/bootstrap.sh   # repo -> live (after a fresh clone)
#   diff /workspace/bootstrap.sh <repo>/scripts/bootstrap.sh && echo in-sync
#
# Keep this file standalone: no repo paths, no secrets. It is committed to a public repo.

export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_XET=1
export SNAP=/workspace/hf/hub/models--maius--llama-3.1-8b-it-personas/snapshots/318b5f7e1428097a1a61d5f0ed205ee048b3f620

# keep pip's temp/cache off the tiny container disk
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pipcache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# python-version-scoped libs: 3.12 and 3.13 images must not share
PY_TAG=$(python3 -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')
export PYLIBS=/workspace/pylibs-$PY_TAG
export PYTHONPATH=$PYLIBS:$PYTHONPATH
export PATH=$PYLIBS/bin:$PATH

if [ ! -d "$PYLIBS/huggingface_hub" ]; then
  echo "installing lite deps to $PYLIBS ..."
  pip install --target="$PYLIBS" -U huggingface_hub hf_transfer
fi

# sox: required by claude's /voice for audio capture. Lives on the CONTAINER disk, not the
# volume, so it has to be reinstalled on every new pod -- hence here rather than one-off.
# The command -v guard keeps this a no-op on shells after the first, since ~/.bashrc sources
# this file every time and apt-get would otherwise run on each new shell.
if ! command -v sox >/dev/null 2>&1; then
  echo "installing sox (for /voice) ..."
  DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq sox libsox-fmt-all >/dev/null 2>&1 \
    || echo "  sox install failed -- /voice will be unavailable"
fi

grep -q 'source /workspace/bootstrap.sh' ~/.bashrc 2>/dev/null || \
  echo 'source /workspace/bootstrap.sh' >> ~/.bashrc

echo "HF_HOME=$HF_HOME | PYLIBS=$PYLIBS"
# git ssh key: volume can't hold 600 perms, stage a copy on container disk
if [ -f /workspace/.ssh/id_ed25519_github ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  cp /workspace/.ssh/id_ed25519_github ~/.ssh/ 2>/dev/null
  chmod 600 ~/.ssh/id_ed25519_github
  # accept-new: every reconnect is a NEW pod with an empty ~/.ssh/known_hosts, so the first
  # git push of each session died on "Host key verification failed". accept-new trusts a host
  # only on first contact and then pins it -- unlike StrictHostKeyChecking=no, it still refuses
  # if a known host's key later CHANGES, which is the case that actually matters.
  export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new"
fi
export CLAUDE_CONFIG_DIR=/workspace/.claude
export PATH=$HOME/.local/bin:/workspace/npm-global/bin:$PATH
command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash
export CLAUDE_CODE_TMPDIR=/tmp/claude-${USER:-$(id -un)}
mkdir -p "$CLAUDE_CODE_TMPDIR"

# tmux, so a dropped SSH connection does not take a running session with it. The container
# is rebuilt on every pod migration and apt packages go with it, so install idempotently
# here rather than once by hand. The config lives on the network volume and survives.
command -v tmux >/dev/null || apt-get install -y -qq tmux >/dev/null 2>&1 || true
export TMUX_CONF=/workspace/.tmux.conf
alias tm='tmux -f "$TMUX_CONF" new-session -A -s main'
