"""Local web dashboard:  propedge web  ->  http://localhost:8000

Runs entirely on your machine (binds to 127.0.0.1). Your Odds API key stays local:
it is read from $ODDS_API_KEY or ~/.propedge/config.json (set from the page).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import threading
import traceback
import hashlib
import hmac
import secrets
import time
from urllib.parse import parse_qs as _pq
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import tracker
from .dist import fit_mean, fit_mean_phi
from .markets import norm_name, phi_prior, resolve
from .model import GameLogs
from .odds import devig, parse_odds
from .scanner import DEFAULT_SHARP, load_quotes

from .paths import home as _home

HOME = _home()
CONFIG = os.path.join(HOME, "config.json")
STATIC = os.path.join(os.path.dirname(__file__), "static", "index.html")

SPORTS = {
    "basketball_nba": {"label": "NBA", "league": "nba",
                       "markets": ["player_points", "player_rebounds", "player_assists", "player_threes",
                                   "player_points_rebounds_assists"]},
    "basketball_wnba": {"label": "WNBA", "league": "wnba",
                        "markets": ["player_points", "player_rebounds", "player_assists"]},
}
LABELS = {"player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists",
          "player_threes": "3-Pointers", "player_points_rebounds_assists": "Pts + Reb + Ast",
          "player_points_rebounds": "Pts + Reb", "player_points_assists": "Pts + Ast",
          "player_rebounds_assists": "Reb + Ast", "player_blocks": "Blocks", "player_steals": "Steals",
          "player_turnovers": "Turnovers"}

_lock = threading.Lock()
AUTH: dict = {"password": None, "secret": b""}
COOKIE = "pe_session"
SESSION_DAYS = 30

LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>propedge — sign in</title>
<style>:root{--bg:#111214;--panel:#17181b;--line:#2a2c31;--text:#ececef;--muted:#9a9ca5;--accent:#5b7cfa}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
body{display:grid;place-items:center;padding:16px;box-sizing:border-box}
form{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:26px;width:min(360px,100%);box-sizing:border-box}
h1{font-size:20px;margin:0 0 4px}p{color:var(--muted);margin:0 0 18px;font-size:13px}
input,button{width:100%;box-sizing:border-box;font:inherit;border-radius:9px;padding:11px 12px}
input{background:#0d0e10;border:1px solid var(--line);color:var(--text);margin-bottom:12px}
button{background:var(--accent);border:0;color:#fff;font-weight:600}.err{color:#ff8a8c;font-size:13px;margin:-4px 0 12px}</style></head>
<body><form method="post" action="/login"><h1>Player Prop Edge</h1><p>Enter your dashboard password.</p>
__ERR__<input type="password" name="password" autofocus autocomplete="current-password" placeholder="Password">
<button>Sign in</button></form></body></html>"""


def _sign(exp: int) -> str:
    return hmac.new(AUTH["secret"], str(exp).encode(), hashlib.sha256).hexdigest()


def _valid_cookie(raw: str | None) -> bool:
    if not raw:
        return False
    for part in raw.split(";"):
        k, _, v = part.strip().partition("=")
        if k == COOKIE and "." in v:
            exp, sig = v.split(".", 1)
            if exp.isdigit() and int(exp) > time.time() and hmac.compare_digest(sig, _sign(int(exp))):
                return True
    return False
_logs_cache: dict = {}


def odds_path(sport):
    return os.path.join(HOME, f"odds_{sport}.json")


def logs_path(sport):
    return os.path.join(HOME, f"logs_{SPORTS.get(sport, {}).get('league', sport)}.csv")


def load_config() -> dict:
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict):
    os.makedirs(HOME, exist_ok=True)
    with open(CONFIG, "w") as f:
        json.dump(cfg, f)
    os.chmod(CONFIG, 0o600)


def _mtime(p):
    return dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="minutes") if os.path.exists(p) else None


def get_logs(sport):
    p = logs_path(sport)
    if not os.path.exists(p):
        return None
    m = os.path.getmtime(p)
    if _logs_cache.get(p, (None,))[0] != m:
        _logs_cache[p] = (m, GameLogs.from_csv(p))
    return _logs_cache[p][1]


def status(sport):
    from . import odds_api
    cfg = load_config()
    return {"sports": {k: v["label"] for k, v in SPORTS.items()},
            "has_key": bool(os.environ.get("ODDS_API_KEY") or cfg.get("odds_api_key")),
            "odds_updated": _mtime(odds_path(sport)), "logs_updated": _mtime(logs_path(sport)),
            "quota": odds_api.LAST_QUOTA or cfg.get("quota"), "markets": SPORTS[sport]["markets"],
            "labels": LABELS}


def build_board(sport: str) -> dict:
    p = odds_path(sport)
    if not os.path.exists(p):
        return {"props": [], "message": "No odds yet. Click “Refresh odds”."}
    quotes = load_quotes(p)
    logs = get_logs(sport)
    groups: dict = {}
    for q in quotes:
        groups.setdefault((q.event, norm_name(q.player), q.market), []).append(q)
    props = []
    for (event, pkey, market), qs in groups.items():
        pairs: dict = {}
        for q in qs:
            pairs.setdefault((q.book, q.line), {})[q.side] = q.price
        obs = []
        for (book, line), sd in pairs.items():
            if "over" in sd and "under" in sd:
                po = devig([1 / sd["over"], 1 / sd["under"]], "power")[0]
                obs.append((book, line, po))
        if not obs:
            continue
        sharp_obs = [o for o in obs if o[0] in DEFAULT_SHARP]
        basis = sharp_obs or obs
        if not sharp_obs and len({o[0] for o in obs}) < 2:
            continue
        phi = fit_mean_phi([(l, p_, 1.0) for _, l, p_ in obs], phi_prior(market, sport))
        mean = fit_mean([(l, p_, 1.0) for _, l, p_ in basis], phi)
        lines = [o[1] for o in sharp_obs] or [l for _, l, _ in obs]
        default_line = max(set(lines), key=lines.count)
        q0 = qs[0]
        row = {"id": f"{event}|{pkey}|{market}", "event": event, "commence": q0.commence,
               "home": q0.home, "away": q0.away, "player": q0.player, "market": market,
               "label": LABELS.get(market, market), "line": default_line, "mean": mean, "phi": phi,
               "sharp": bool(sharp_obs), "sharp_books": sorted({o[0] for o in sharp_obs}),
               "quotes": [{"book": q.book, "line": q.line, "side": q.side, "price": q.price} for q in qs],
               "team": None, "opp": None, "pos": "", "athlete_id": "", "games": [], "model_mean": None}
        if logs is not None:
            games = logs.by_player.get(pkey, [])
            if games:
                team = games[-1].get("team")
                row["team"] = team
                row["pos"] = games[-1].get("pos") or ""
                aid = games[-1].get("athlete_id")
                row["athlete_id"] = str(int(aid)) if isinstance(aid, float) else (aid or "")
                if team == q0.home:
                    row["opp"] = q0.away
                elif team == q0.away:
                    row["opp"] = q0.home
                _, cols, _ = resolve(market, sport)
                for g in games:
                    v = GameLogs._stat(g, cols)
                    if v is not None:
                        row["games"].append([g["date"], g.get("opponent", ""), v, g.get("minutes")])
                try:
                    proj = logs.project(q0.player, market, opponent=row["opp"], sport=sport)
                    row["model_mean"] = proj.mean if proj else None
                except KeyError:
                    pass
        props.append(row)
    return {"props": props, "odds_updated": _mtime(p), "books": sorted({q.book for q in quotes})}


def refresh_odds(sport, markets, max_events, hours):
    from .odds_api import LAST_QUOTA, fetch_props
    events = fetch_props(sport, markets, "us,us2,eu", None, max_events, hours)
    os.makedirs(HOME, exist_ok=True)
    with open(odds_path(sport), "w") as f:
        json.dump(events, f)
    cfg = load_config()
    if LAST_QUOTA:
        cfg["quota"] = dict(LAST_QUOTA)
        save_config(cfg)
    return {"events": len(events), "quota": LAST_QUOTA}


def refresh_logs(sport):
    from .espn import fetch_logs, write_csv
    league = SPORTS[sport]["league"]
    today = dt.date.today()
    m = 10 if league == "nba" else 5
    y = today.year if today.month >= m else today.year - 1
    rows = fetch_logs(dt.date(y, m, 1), today - dt.timedelta(days=1), league, verbose=False)
    os.makedirs(HOME, exist_ok=True)
    write_csv(rows, logs_path(sport))
    return {"rows": len(rows)}


def bets_payload():
    con = tracker.connect()
    rows = [dict(r) for r in tracker.list_bets(con)]
    return {"bets": rows, "report": tracker.report(con)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _authed(self) -> bool:
        return not AUTH["password"] or _valid_cookie(self.headers.get("Cookie"))

    def _redirect(self, to, cookie=None):
        self.send_response(303)
        self.send_header("Location", to)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _secure(self) -> bool:
        return self.headers.get("X-Forwarded-Proto", "") == "https"

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        oh = urlparse(origin).hostname
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return oh in ("localhost", "127.0.0.1") or oh == host

    def _guard(self, fn):
        try:
            self._send(200, fn())
        except SystemExit as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # surface errors to the page
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        sport = q.get("sport", "basketball_nba")
        if u.path == "/login":
            return self._send(200, LOGIN_HTML.replace("__ERR__", "").encode(), "text/html; charset=utf-8")
        if u.path == "/logout":
            return self._redirect("/login", f"{COOKIE}=; Max-Age=0; Path=/")
        if u.path == "/manifest.webmanifest":
            return self._send(200, MANIFEST, "application/manifest+json")
        if not self._authed():
            if u.path.startswith("/api/"):
                return self._send(401, {"error": "not signed in"})
            return self._redirect("/login")
        if u.path in ("/", "/index.html"):
            with open(STATIC, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/status":
            return self._guard(lambda: status(sport))
        if u.path == "/api/board":
            return self._guard(lambda: build_board(sport))
        if u.path == "/api/bets":
            return self._guard(bets_payload)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        # Only accept requests from this page (blocks other websites from poking the server)
        if not self._same_origin():
            return self._send(403, {"error": "forbidden origin"})
        u = urlparse(self.path)
        if u.path == "/login":
            n = int(self.headers.get("Content-Length") or 0)
            pw = _pq(self.rfile.read(n).decode()).get("password", [""])[0]
            if AUTH["password"] and hmac.compare_digest(pw.encode(), AUTH["password"].encode()):
                exp = int(time.time()) + SESSION_DAYS * 86400
                ck = (f"{COOKIE}={exp}.{_sign(exp)}; Max-Age={SESSION_DAYS * 86400}; Path=/; HttpOnly; SameSite=Lax"
                      + ("; Secure" if self._secure() else ""))
                return self._redirect("/", ck)
            time.sleep(1.0)  # slow down password guessing
            err = '<div class="err">Wrong password.</div>'
            return self._send(401, LOGIN_HTML.replace("__ERR__", err).encode(), "text/html; charset=utf-8")
        if not self._authed():
            return self._send(401, {"error": "not signed in"})
        b = self._body()
        sport = b.get("sport", "basketball_nba")
        if u.path == "/api/refresh-odds":
            def go():
                with _lock:
                    return refresh_odds(sport, b.get("markets") or SPORTS[sport]["markets"],
                                        b.get("max_events") or None, float(b.get("hours") or 36))
            return self._guard(go)
        if u.path == "/api/refresh-logs":
            return self._guard(lambda: refresh_logs(sport))
        if u.path == "/api/config":
            def go():
                cfg = load_config()
                cfg["odds_api_key"] = (b.get("odds_api_key") or "").strip()
                save_config(cfg)
                return {"ok": True}
            return self._guard(go)
        if u.path == "/api/bets":
            def go():
                con = tracker.connect()
                bid = tracker.add(con, b["player"], b["market"], float(b["line"]), b["side"],
                                  float(b["odds"]), float(b["stake"]), b.get("book", ""),
                                  b.get("event", ""), b.get("fair_p"))
                return {"id": bid}
            return self._guard(go)
        if u.path == "/api/bets/settle":
            return self._guard(lambda: (tracker.settle(tracker.connect(), int(b["id"]), b["result"]), {"ok": True})[1])
        if u.path == "/api/bets/close":
            return self._guard(lambda: (tracker.close(tracker.connect(), int(b["id"]), parse_odds(b["over"]),
                                                      parse_odds(b["under"])), {"ok": True})[1])
        self._send(404, {"error": "not found"})


MANIFEST = {"name": "Player Prop Edge", "short_name": "PropEdge", "start_url": "/", "display": "standalone",
            "background_color": "#111214", "theme_color": "#111214"}


def _lan_ip() -> str | None:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packets sent; just picks the outbound interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def serve(port: int = 8000, host: str = "127.0.0.1", open_browser: bool = True, password: str | None = None):
    password = password or os.environ.get("PROPEDGE_PASSWORD")
    local_only = host in ("127.0.0.1", "localhost", "::1")
    if not local_only and not password:
        raise SystemExit("Reachable from other devices, so a password is required:\n"
                         "  propedge web --phone --password 'something-long'   (or set PROPEDGE_PASSWORD)")
    if password:
        cfg = load_config()
        if not cfg.get("session_secret"):
            cfg["session_secret"] = secrets.token_hex(32)
            save_config(cfg)
        AUTH.update(password=password, secret=bytes.fromhex(cfg["session_secret"]))
    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://localhost:{port}"
    print(f"propedge dashboard running at {url}  (Ctrl+C to stop)", file=sys.stderr)
    if not local_only:
        ip = _lan_ip()
        if ip:
            print(f"On your phone (same Wi-Fi), open:  http://{ip}:{port}", file=sys.stderr)
    if open_browser:
        try:
            import webbrowser
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
