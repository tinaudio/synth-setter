#!/usr/bin/env bash
tempfile=$(mktemp)
Xvfb -displayfd 3 -screen 0 1280x720x24 -ac 3>"$tempfile" &
XVFB_PID=$!

MAX_WAIT=5
INTERVAL=0.1
TIME_SPENT=0

while [ ! -s "$tempfile" ]; do
sleep $INTERVAL
TIME_SPENT=$(awk -v t1="$TIME_SPENT" -v t2="$INTERVAL" 'BEGIN { print t1 + t2 }')
if (( $(awk -v ts="$TIME_SPENT" -v mw="$MAX_WAIT" 'BEGIN { print (ts > mw) }') )); then
    echo "ERROR: Timeout waiting for Xvfb to write display number into $tempfile."
    exit 1
fi
done

DISPLAY_NAME=$(cat "$tempfile")
echo "Xvfb chose display: $DISPLAY_NAME"
export DISPLAY=:$DISPLAY_NAME

if [ "$3" = "simple" ]
then
    echo "Using SIMPLE dataset"
    SPEC=surge_simple
    PRESET=surge-simple
elif [ "$3" = "ffn_full" ]
then
    echo "Using FULL dataset with ffn"
    SPEC=surge_xt
    PRESET=surge-base
elif [ "$3" = "ffn" ]
then
    SPEC=surge_simple
    PRESET=surge-simple
elif [ "$3" = "flowmlp_full" ]
then
    echo "Using FULL dataset with flowmlp"
    SPEC=surge_xt
    PRESET=surge-base
elif [ "$3" = "flowmlp" ]
then
    echo "Using SIMPLE dataset with flowmlp"
    SPEC=surge_simple
    PRESET=surge-simple
else
    echo "Using FULL dataset"
    SPEC=surge_xt
    PRESET=surge-base
fi


openbox &
xsettingsd &
python scripts/predict_vst_audio.py -X -S -r presets/$PRESET.vstpreset --param_spec $SPEC $1 $2
kill $XVFB_PID
rm "$tempfile"
