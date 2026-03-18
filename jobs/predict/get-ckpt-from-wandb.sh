#!/usr/bin/env bash
wandbid=$1
CANDIDATES=$(
  find logs/train -type d -wholename "*wandb/*$wandbid" -print0 \
  | xargs -0 ls -dt
)

# 2. Loop through each candidate directory in descending time order.
for DIR in $CANDIDATES; do
  CANDIDATE_CKPT_PATH="${DIR%/*/*}/checkpoints/last.ckpt"
  if [ -f "$CANDIDATE_CKPT_PATH" ]; then
    # Found a valid checkpoint, so set variables and break.
    WANDB_DIR="$DIR"
    CKPT_PATH="$CANDIDATE_CKPT_PATH"
    break
  fi
done

# Optional: Check if no valid checkpoint was found and handle accordingly
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
  echo "No valid checkpoint found for pattern '$wandbid'."
  exit 1
fi
