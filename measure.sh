#!/bin/bash
python main.py &
PY_PID=$!
nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > power_log-$PY_PID.csv &
POWER_PID=$!
nvidia-smi pmon -s um -d 1 -o DT > gpu_utilization-$PY_PID.log &
PMON_PID=$!
wait $PY_PID
kill $POWER_PID $PMON_PID 2>/dev/null
