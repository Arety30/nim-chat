#!/usr/bin/env bash
# Deploy nim-chat to cloud-vm as https://example.com (Apache :443 -> uvicorn 127.0.0.1:8000).
set -euo pipefail
VM=${VM:-cloud-vm}
HERE=$(cd "$(dirname "$0")" && pwd)
rsync -az --delete --exclude .venv --exclude __pycache__ "$HERE/" "$VM:~/nim-chat/"
ssh "$VM" 'mkdir -p ~/.config/nim-chat && chmod 700 ~/.config/nim-chat'
scp -q ~/.config/nim-chat/env "$VM:~/.config/nim-chat/env"
ssh "$VM" bash -s <<'REMOTE'
set -euo pipefail
H=$HOME; C=$H/cloud
chmod 600 ~/.config/nim-chat/env
sed -i 's/^HOST=.*/HOST=127.0.0.1/; s/^PORT=.*/PORT=8000/' ~/.config/nim-chat/env
# venv
if [ ! -x $H/nim-chat/.venv/bin/python ]; then python3 -m venv $H/nim-chat/.venv; fi
$H/nim-chat/.venv/bin/pip install -q -r $H/nim-chat/requirements.txt
$H/nim-chat/.venv/bin/pip install -q -r $H/nim-chat/requirements-voice.txt || echo "(faster-whisper install failed; browser STT still works)"
# supervisord: park stt-web, add nim-chat
if ! grep -q '^\[program:nim-chat\]' $C/supervisord.conf; then
  python3 - "$C/supervisord.conf" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p).read()
# stt-web -> autostart=false (kept for rollback; it shares port 8000)
m=re.search(r'\[program:stt-web\].*?(?=\n\[|\Z)', s, re.S)
if m and 'autostart=false' not in m.group(0):
    block=m.group(0).rstrip('\n')+'\nautostart=false\n'
    s=s[:m.start()]+block+s[m.end():]
s=s.rstrip('\n')+'''

[program:nim-chat]
; NVIDIA NIM chat + agent harness (~/nim-chat), served as https://example.com via Apache (~/cloud/tests-arety-cloud.conf). Replaced stt-web on 2026-09-12.
directory=/home/USER/nim-chat
command=/home/USER/nim-chat/.venv/bin/python server.py
environment=HOST="127.0.0.1",PORT="8000"
autorestart=true
startsecs=5
stdout_logfile=/home/USER/cloud/logs/nim-chat.log
redirect_stderr=true
'''
open(p,'w').write(s)
PY
fi
# boot script: recreate venv after a reset if it is ever lost
if ! grep -q 'nim-chat/.venv' $C/user-boot.sh; then
  python3 - "$C/user-boot.sh" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
add='''if [ ! -x $H/nim-chat/.venv/bin/python ]; then
  python3 -m venv $H/nim-chat/.venv && $H/nim-chat/.venv/bin/pip install -q -r $H/nim-chat/requirements.txt
fi
'''
s=s.replace('# --- supervisord (user instance)', add+'# --- supervisord (user instance)',1)
open(p,'w').write(s)
PY
fi
# apache: longer proxy timeout for long tool runs / slow first token
if ! grep -q ProxyTimeout $C/tests-arety-cloud.conf; then
  sed -i 's|^# https://example.com -> STT web app.*|# https://example.com -> nim-chat (NVIDIA NIM chat + agent harness, ~/nim-chat, uvicorn 127.0.0.1:8000).|; s|    ProxyPassReverse / http://127.0.0.1:8000/|    ProxyPassReverse / http://127.0.0.1:8000/\n    ProxyTimeout 900|' $C/tests-arety-cloud.conf
  sudo cp $C/tests-arety-cloud.conf /etc/apache2/sites-available/tests-arety-cloud.conf
  sudo apachectl -t && sudo apachectl graceful
fi
supervisorctl -c $C/supervisord.conf stop stt-web >/dev/null || true
supervisorctl -c $C/supervisord.conf reread >/dev/null
supervisorctl -c $C/supervisord.conf update >/dev/null
supervisorctl -c $C/supervisord.conf restart nim-chat >/dev/null || supervisorctl -c $C/supervisord.conf start nim-chat
sleep 7
supervisorctl -c $C/supervisord.conf status nim-chat stt-web
curl -s -m 5 http://127.0.0.1:8000/healthz; echo
tail -n 3 $C/logs/nim-chat.log
REMOTE
echo "public check:"
curl -s -m 20 -o /dev/null -w "https://example.com -> %{http_code}\n" https://example.com/healthz
curl -s -m 20 https://example.com/healthz; echo
