"""Run commands on a RunPod pod over the ssh.runpod.io proxy.

Why not plain `ssh host "cmd"`: the proxy ignores a remote command and always
opens an interactive shell, and it refuses a session without a PTY. So the
commands go in on stdin with `-tt`, and the echoed prompt/ANSI noise is
stripped back out here.

    python podsh.py <podId> "cmd1" "cmd2" ...
    python podsh.py <podId> --file script.sh      # run a local script remotely

Exit status is the remote script's, carried out through a sentinel line, so a
failing remote command fails this process too.
"""
import base64
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"
KEY = re.search(r"apikey\s*=\s*['\"]?([^'\"\s]+)['\"]?",
                (Path.home() / ".runpod" / "config.toml").read_text()
                ).group(1).strip("'\"")
IDENTITY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
DONE = "__PODSH_RC__"


def gql(query: str) -> dict:
    req = urllib.request.Request(
        f"{API}?api_key={KEY}", data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "runpodctl/2.9.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def host_id(pod: str) -> str:
    d = gql('query { pod(input:{podId:"%s"}) { machine { podHostId } } }' % pod)
    h = (((d.get("data") or {}).get("pod") or {}).get("machine") or {}).get("podHostId")
    if not h:
        raise SystemExit(f"no podHostId for {pod}: {json.dumps(d)[:300]}")
    return h


ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07?")


def run(pod: str, script: str, timeout: int = 900) -> tuple[int, str]:
    # The proxy echoes every line back and the PTY hard-wraps at the terminal
    # width, inserting a line break mid-token -- which does not just make the
    # transcript ugly, it CORRUPTS any command longer than the width, because
    # the shell reads the wrapped text. So nothing long is ever typed: the
    # script goes over as short base64 chunks appended to a file, and only
    # then is it decoded and run.
    b64 = base64.b64encode(script.encode()).decode()
    chunks = [b64[i:i + 180] for i in range(0, len(b64), 180)]
    lines = [": > /tmp/podsh.b64"]
    lines += [f"printf %s {c} >> /tmp/podsh.b64" for c in chunks]
    lines += ["base64 -d /tmp/podsh.b64 > /tmp/podsh.sh",
              "bash /tmp/podsh.sh"]
    payload = "\n".join(lines) + f"\nprintf '{DONE}%s\\n' \"$?\"\nexit\n"
    p = subprocess.run(
        ["ssh", "-tt", "-i", str(IDENTITY),
         "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=30",
         f"{host_id(pod)}@ssh.runpod.io"],
        input=payload, capture_output=True, text=True, timeout=timeout)
    out = ANSI.sub("", (p.stdout or "") + (p.stderr or "")).replace("\r", "")
    rc = 0
    for line in out.splitlines():
        if line.startswith(DONE) and line[len(DONE):].strip().isdigit():
            rc = int(line[len(DONE):].strip())
    body = "\n".join(l for l in out.splitlines()
                     if DONE not in l and "ssh.runpod.io closed" not in l)
    return rc, body


def main() -> None:
    pod, rest = sys.argv[1], sys.argv[2:]
    timeout = 900
    if "--timeout" in rest:
        i = rest.index("--timeout")
        timeout = int(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    if rest and rest[0] == "--file":
        script = Path(rest[1]).read_text(encoding="utf-8")
    else:
        script = "\n".join(rest)
    rc, body = run(pod, script, timeout)
    print(body)
    sys.exit(rc)


if __name__ == "__main__":
    main()
