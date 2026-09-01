for i in $(seq 1 720); do
  if grep -q "all done" results/logs/run_s1_remaining.log 2>/dev/null; then
    echo "DONE"
    exit 0
  fi
  if grep -q "Traceback" results/logs/run_s1_remaining.log 2>/dev/null; then
    echo "ERROR IN LOG"
    exit 1
  fi
  if ! ps -ef 2>/dev/null | grep -q "[r]un_a --model"; then
    echo "PROCESS EXITED (check log)"
    exit 1
  fi
  sleep 30
done
echo "TIMEOUT after 6h"
exit 2
