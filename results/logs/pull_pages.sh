# Page ai.AgnoSessions through the database skill CLI (credentials stay inside
# the skill) into data/real_sessions/pages/*.json (gitignored: PII).
# Run from the repo root in Git Bash:  bash results/logs/pull_pages.sh
# Then: python -m harness.real_requests extract && python -m harness.real_requests histogram
cd /c/code/covenant-agent
mkdir -p data/real_sessions/pages
PAGE=20; off=0; n=0
while true; do
  f=$(printf "data/real_sessions/pages/page_%04d.json" $off)
  powershell -Command "cd C:\Users\josha\.claude\skills\database\sql-server-cli; node dist/cli/index.js read-data \"SELECT id, accountId, agentId, sessionId, userId, model, numUserPrompts, CONVERT(varchar(33), __createdAt, 127) AS createdAt, runs FROM ai.AgnoSessions WHERE __deleted = 0 ORDER BY __createdAt OFFSET $off ROWS FETCH NEXT $PAGE ROWS ONLY\"" 2>&1 | grep -v "^Executing" > "$f"
  got=$(python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('recordCount',0))" "$f" 2>/dev/null || echo ERR)
  echo "offset $off -> $got"
  if [ "$got" = "ERR" ]; then echo "PAGE ERROR at $off (retrying in 10s)"; sleep 10; continue; fi
  if [ "$got" = "0" ]; then rm -f "$f"; break; fi
  off=$((off+PAGE)); n=$((n+got))
done
echo "=== all done: $n sessions ==="
