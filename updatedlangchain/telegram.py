#!/usr/bin/env python3
"""
tg_snipe.py - make a Telegram message ARRIVE at Telegram's servers at an exact moment.

Why you keep coming 2nd/3rd: your PC clock is usually off by 20-500 ms, and the
message also needs ~30-100 ms to travel to Telegram. This script fixes both:
  1. syncs to atomic time via NTP (millisecond accuracy, no admin rights needed)
  2. measures the round-trip to YOUR Telegram datacenter and fires early by the
     one-way latency, so the message lands at 10:03:00.000 + margin
  3. pre-connects, pre-resolves the chat, keeps the connection warm, and
     busy-waits the last few ms so nothing is slow at the critical moment
  4. if the group is still locked at that instant, it re-sends back-to-back
     until it goes through

Setup (one time):
    pip install telethon cryptg qrcode
    Get api_id / api_hash at https://my.telegram.org  ->  "API development tools"
    Fill in the CONFIG section below OR a .env file next to this script.
    First run asks for your phone number + login code; a session file is saved.

Step 1 - calibrate (sends ~24 test messages to your Saved Messages, then deletes them):
    python tg_snipe.py --calibrate

Step 2 - real run (start it any time before 10:02:20):
    python tg_snipe.py

Priority of settings:  command-line flag  >  .env / environment variable  >  CONFIG below.
"""
import argparse
import asyncio
import datetime as dt
import math
import os
import random
import socket
import statistics
import struct
import sys
import time

# ==========================================================================
# CONFIG - fill these in here, or put the same keys in a .env file
# (values in .env / environment variables override what is written here)
# ==========================================================================
CONFIG = {
    "TG_API_ID":     "YOUR_API_ID",        # e.g. 12345678   (from my.telegram.org)
    "TG_API_HASH":   "YOUR_API_HASH",      # e.g. 0123456789abcdef0123456789abcdef
    "TG_PHONE":      "YOUR_PHONE",         # e.g. +919876543210  (used only for first login)
    "TG_SESSION":    "snipe",              # session file name (snipe.session is created)
    "TG_CHAT":       "YOUR_CHAT",          # @group_username, https://t.me/xyz, -100123..., or @some_bot
    "TG_TEXT":       "YOUR_MESSAGE",       # the exact message (single line; use \n for new lines)
    "TG_TEXT_FILE":  "message.txt",        # file(s) with the message; several: "msg_151.txt,msg_156.txt"
    "TG_AT":         "10:03:00",           # target time in IST, HH:MM:SS
    "TG_MARGIN_MS":  "10",                 # ms AFTER target to land (use value from --calibrate)
    "TG_RETRY_FOR":  "3",
    "TG_FIRE_OFFSET_MS": "",               # leave empty: --calibrate saves this to calibration.json
    "TG_CALIBRATE_CHAT": "me",
    "TG_BURST":      "12",                 # copies to send around 10:03:00 (1 = single message)
    "TG_BURST_SPREAD_MS": "600",           # window the copies are spread over (ms)
    "TG_BURST_EARLY_MS": "0",             # start this many ms before the estimated edge
    "TG_DELETE_EXTRAS": "0",               # 0 = keep every copy, 1 = delete all but first on-time
    "TG_REFERENCE":  "realtime",           # realtime = real 10:03:00 like everyone's phone; telegram = TG stamps
    "TG_LIVE_CALIBRATION": "1",            # 1 = auto-calibrate on Telegram's clock 50 s before target             # chat for calibration tests (me = Saved Messages)                  # seconds to keep re-sending if chat is still locked
}


def _load_env_file(path):
    """Minimal .env reader (no extra package needed). KEY=value per line."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.lstrip()
        if v[:1] in ("'", '"'):
            q = v[0]
            v = v[1:]
            # multi-line quoted value: keep reading until the closing quote
            while not v.rstrip().endswith(q) and i < len(lines):
                v += "\n" + lines[i]
                i += 1
            v = v.rstrip()
            if v.endswith(q):
                v = v[:-1]
        else:
            v = v.strip()
        os.environ.setdefault(k, v)     # real env vars still win


_load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
_load_env_file(".env")


def cfg(key):
    v = os.environ.get(key, CONFIG.get(key, ""))
    return None if (v == "" or str(v).startswith("YOUR_")) else v

try:  # faster event loop on Linux/macOS (optional)
    import uvloop
    uvloop.install()
except Exception:
    pass

from telethon import TelegramClient, functions, types
from telethon.errors import FloodWaitError, RPCError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# --------------------------------------------------------------------------
# Precise clock: perf_counter (sub-microsecond) anchored to wall time + NTP fix
# --------------------------------------------------------------------------
def _anchor():
    t = time.time()
    while time.time() == t:        # wait for a tick so the anchor is exact
        pass
    return time.time(), time.perf_counter()

_BASE_WALL, _BASE_PERF = _anchor()
_OFFSET = 0.0                       # correction from NTP (seconds)


def _raw():
    return _BASE_WALL + (time.perf_counter() - _BASE_PERF)


def now():
    """Best estimate of true UTC epoch seconds."""
    return _raw() + _OFFSET


def fmt(t):
    return dt.datetime.fromtimestamp(t, IST).strftime("%H:%M:%S.%f")[:-3]


NTP_SERVERS = ["time.google.com", "time.cloudflare.com", "time.windows.com", "pool.ntp.org"]


def _sntp(server, timeout=0.6):
    addr = socket.getaddrinfo(server, 123, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        t0 = _raw()
        s.sendto(b"\x23" + b"\0" * 47, addr)
        data, _ = s.recvfrom(48)
        t3 = _raw()

    def ts(i):
        sec, frac = struct.unpack("!II", data[i:i + 8])
        return sec - 2208988800 + frac / 2**32

    t1, t2 = ts(32), ts(40)
    return ((t1 - t0) + (t2 - t3)) / 2, (t3 - t0) - (t2 - t1)   # offset, delay


def sync_clock(samples=4):
    """Returns (offset_s, best_delay_s) or None. Uses lowest-delay samples."""
    global _OFFSET
    res = []
    for srv in NTP_SERVERS:
        for _ in range(samples):
            try:
                res.append(_sntp(srv))
            except Exception:
                pass
    if not res:
        return None
    res.sort(key=lambda r: r[1])
    _OFFSET = statistics.median(r[0] for r in res[:3])
    return _OFFSET, res[0][1]


# --------------------------------------------------------------------------
# Telegram helpers
# --------------------------------------------------------------------------
def _ping():
    return functions.PingRequest(ping_id=random.getrandbits(62))


async def measure_rtt(client, n=15):
    rtts = []
    for _ in range(n):
        t0 = time.perf_counter()
        await client(_ping())
        rtts.append(time.perf_counter() - t0)
        await asyncio.sleep(0.05)
    return rtts


async def wait_until(t, client=None):
    """Sleep until t (true time); keep connection warm; spin the last 150 ms."""
    while True:
        d = t - now()
        if d <= 0.15:
            break
        if client is not None and d > 2.5:
            await client(_ping())                       # keep socket hot
            await asyncio.sleep(min(5.0, d - 2.0))
        else:
            await asyncio.sleep(d - 0.15)
    while now() < t:                                    # busy-wait final ms
        pass


def send_req(peer, text):
    return functions.messages.SendMessageRequest(
        peer=peer, message=text, no_webpage=True, random_id=random.getrandbits(63))


def extract(res):
    if isinstance(res, types.UpdateShortSentMessage):
        return res.id, res.date
    for u in getattr(res, "updates", []):
        if isinstance(u, (types.UpdateNewMessage, types.UpdateNewChannelMessage)):
            return u.message.id, u.message.date
    return None, None


async def prepare(client, ntp=True):
    r = "skip"
    if ntp:
        print("Syncing clock with NTP ...")
        r = await asyncio.get_running_loop().run_in_executor(None, sync_clock)
    if r == "skip":
        print("Skipping NTP - timing is measured directly against Telegram's clock instead.")
    elif r is None:
        print("  !! NTP blocked on this network. Using system clock - sync it manually "
              "(Windows: Settings > Time > 'Sync now').")
    else:
        print(f"  your clock is off by {r[0]*1000:+.1f} ms (corrected; NTP delay {r[1]*1000:.1f} ms)")
    print("Measuring latency to Telegram ...")
    rtts = await measure_rtt(client)
    med, lo, hi = statistics.median(rtts), min(rtts), max(rtts)
    print(f"  RTT median {med*1000:.1f} ms  (min {lo*1000:.1f}, max {hi*1000:.1f})")
    if hi - lo > 0.04:
        print("  !! High jitter. Use wired internet / close downloads, or run from a VPS near your DC.")
    LAST_RTTS[:] = rtts
    return med / 2


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
LAST_RTTS = []


async def measure_q(client, peer, text, n=12):
    """
    Send n test messages at different points inside a second and read Telegram's own
    timestamps. Returns (q_lo, q_hi, ids): pressing send at local time t gets Telegram
    stamp floor(t + Q), with Q in [q_lo, q_hi]. Independent of NTP / PC clock errors.
    """
    ids, samples = [], []
    for k in range(n):
        S = math.floor(now()) + 2
        await wait_until(S + k / n)
        t_send = now()
        res = await client(send_req(peer, text))
        mid, date = extract(res)
        if mid:
            ids.append(mid)
        if date:
            st = date.timestamp()
            samples.append((st - t_send, st + 1 - t_send))
    lo_all = min(a for a, _ in samples)
    hi_all = max(b for _, b in samples)
    grid = [lo_all + i / 1000 for i in range(int((hi_all - lo_all) * 1000) + 1)]
    counts = [sum(1 for a, b in samples if a <= g < b) for g in grid]
    best = max(counts)
    good = [g for g, c in zip(grid, counts) if c == best]
    print(f"  Telegram delay+clock difference Q = {min(good)*1000:+.0f} .. {max(good)*1000:+.0f} ms "
          f"({best}/{len(samples)} test sends agree)")
    return min(good), max(good), ids


async def calibrate(client, text=None, chat="me"):
    """
    Measures Q = time from pressing send (on your clock) to the moment Telegram stamps
    the message (on TELEGRAM's clock). This includes network delay AND any difference
    between Telegram's clock and yours, so no assumptions are needed.
    Then fine-tunes with real test sends. Uses your REAL message text.
    """
    import json
    await prepare(client)
    peer = await client.get_input_entity(chat)
    text = text or "calibration"
    ids = []

    async def send_at(t):
        await wait_until(t)
        t_send = now()
        res = await client(send_req(peer, text))
        mid, date = extract(res)
        if mid:
            ids.append(mid)
        return t_send, date.timestamp()

    try:
        # ---- Step 1: Telegram stamps whole seconds; sending at 20 different points
        # inside a second pins down Q (each send says Q is in [stamp-t, stamp+1-t)).
        N = 20
        print(f"\nStep 1/2 - measuring the send->Telegram-timestamp delay ({N} test sends) ...")
        samples = []
        for k in range(N):
            S = math.floor(now()) + 2
            t_send, stamp = await send_at(S + k / N)
            samples.append((stamp - t_send, stamp + 1 - t_send))
        lo_all = min(a for a, _ in samples)
        hi_all = max(b for _, b in samples)
        grid = [lo_all + i / 1000 for i in range(int((hi_all - lo_all) * 1000) + 1)]
        counts = [sum(1 for a, b in samples if a <= g < b) for g in grid]
        best = max(counts)
        good = [g for g, c in zip(grid, counts) if c == best]
        Q = (min(good) + max(good)) / 2
        print(f"  Q = {Q*1000:+.0f} ms   ({best}/{N} sends agree, range "
              f"{min(good)*1000:+.0f}..{max(good)*1000:+.0f} ms)")
        if LAST_RTTS:
            skew = Q - statistics.median(LAST_RTTS) / 2
            print(f"  (=> Telegram's timestamps run roughly {abs(skew)*1000:.0f} ms "
                  f"{'BEHIND' if skew < 0 else 'AHEAD OF'} your synced clock. The script now "
                  "aims at TELEGRAM's 10:03:00, which is what the group sees.)")
        center = round(-Q * 1000)            # send offset that lands exactly on the second

        # ---- Step 2: fine sweep around that point with real sends
        lo, hi = center - 60, center + 60
        print(f"\nStep 2/2 - fine search from {lo:+d} to {hi:+d} ms ...")
        res_f = {}
        d = lo
        while d <= hi or (res_f and res_f[max(res_f)] > 0 and d <= hi + 200):
            early = 0
            for _ in range(3):
                S = math.floor(now()) + 2
                t_send, stamp = await send_at(S + d / 1000)
                early += int(stamp < S)
            res_f[d] = early
            print(f"  send at {d:+5d} ms : {3-early}/3 on time" + ("" if not early else f"   ({early} EARLY)"))
            d += 10
        safe = [x for x in sorted(res_f) if all(res_f[y] == 0 for y in res_f if y >= x)]
        if not safe:
            print("Could not find a safe point - network too unstable. Try another network and re-run.")
            return
        edge = safe[0]
    finally:
        if ids:
            await client.delete_messages(chat, ids)

    jitter = (max(LAST_RTTS) - min(LAST_RTTS)) * 1000 if LAST_RTTS else 40
    buffer = max(10, round(jitter / 4))
    offset = edge + buffer
    with open(CALIB_FILE, "w") as f:
        json.dump({"fire_offset_ms": offset, "edge_ms": edge, "buffer_ms": buffer,
                   "q_ms": round(Q * 1000), "measured_at": now()}, f)
    print(f"\nEdge: sending at {edge:+d} ms is the earliest that always got the right second.")
    print(f"Saved FIRE OFFSET = {offset:+d} ms (edge + {buffer} ms jitter buffer) to calibration.json")
    print("The real run uses this automatically (valid 12 h). Re-calibrate on the same network "
          "shortly before 10:03.")


async def spin_until(t):
    """Like wait_until but yields to the event loop so queued sends go out immediately."""
    d = t - now()
    if d > 0.03:
        await asyncio.sleep(d - 0.03)
    while now() < t:
        await asyncio.sleep(0)


def _label(text):
    first = text.strip().splitlines()[0] if text.strip() else "(empty)"
    return first if len(first) <= 30 else first[:29] + "~"


async def burst_send(client, peer, a, target, start):
    """
    Send every message in a.texts, a.burst times each, over a.burst_spread_ms from `start`.
    Copies are interleaved (msg1, msg2, msg1, msg2, ...) so each message's first copy
    goes out in the very first round, back-to-back.
    If a.delete_extras: per message keep only the first (on-time) copy, delete the rest.
    """
    texts = a.texts
    n, spread = a.burst, a.burst_spread_ms / 1000
    times = [start + (i * spread / (n - 1) if n > 1 else 0.0) for i in range(n)]
    plan = [(t, k, send_req(peer, texts[k])) for t in times for k in range(len(texts))]
    for _, _, r in plan:
        bytes(r)                                        # pre-serialize everything
    await wait_until(times[0] - 0.2)
    sent = []
    for t, k, r in plan:
        await spin_until(t)                             # same t in a round -> back-to-back
        sent.append((now(), k, asyncio.ensure_future(client(r))))
    results = await asyncio.gather(*(f for _, _, f in sent), return_exceptions=True)

    tsec = math.floor(target)
    rows = []                                           # (t_send, k, mid, stamp, err)
    ok = {k: [] for k in range(len(texts))}
    for (t_send, k, _), res in zip(sent, results):
        if isinstance(res, Exception):
            rows.append((t_send, k, None, None, type(res).__name__))
            continue
        mid, date = extract(res)
        stamp = int(date.timestamp()) if date else None
        rows.append((t_send, k, mid, stamp, ""))
        if mid:
            ok[k].append((mid, stamp))

    keep = {}
    for k, lst in ok.items():
        cands = sorted(lst) if a.realtime else sorted((m, st) for m, st in lst
                                                      if st is not None and st >= tsec)
        keep[k] = cands[0][0] if cands else None
    if a.delete_extras:
        to_delete = [m for k, lst in ok.items() for m, _ in lst if m != keep[k]]
        if to_delete:
            try:
                await client.delete_messages(peer, to_delete, revoke=True)
            except Exception as e:
                print(f"!! Could not delete extra copies ({type(e).__name__}) - delete them manually NOW")

    for k in range(len(texts)):                         # nothing usable -> send one now
        if keep[k] is None:
            print(f"No usable copy of '{_label(texts[k])}' - sending one more immediately ...")
            t_fb = now()
            try:
                res = await client(send_req(peer, texts[k]))
                keep[k], date = extract(res)
                rows.append((t_fb, k, keep[k], int(date.timestamp()) if date else None, "fallback"))
            except RPCError as e:
                rows.append((t_fb, k, None, None, type(e).__name__))

    print("\n===== BURST RESULT =====")
    for t_send, k, mid, stamp, err in rows:
        st = dt.datetime.fromtimestamp(stamp, IST).strftime("%H:%M:%S") if stamp else "--:--:--"
        if not mid:
            tag = f"failed: {err}"
        elif mid == keep[k]:
            tag = "<== FIRST" if a.realtime else "<== FIRST ON-TIME (this one counts)"
        elif a.delete_extras:
            tag = "deleted"
        else:
            tag = ("EARLY (still visible)" if stamp is not None and stamp < tsec and not a.realtime
                   else "extra copy")
        name = f"[{_label(texts[k])}]" if len(texts) > 1 else ""
        print(f"  sent {fmt(t_send)} {name} -> stamp {st}  id {mid}  {tag}")

    print()
    for k in range(len(texts)):
        if keep[k]:
            print(f"  first copy of '{_label(texts[k])}' = message id {keep[k]}")
        else:
            print(f"  '{_label(texts[k])}' FAILED - nothing was sent")
    if a.realtime:
        print("\nSent at real target time (by NTP). This group's Telegram stamps run ~1-2 s behind "
              "real time, so the app may show the previous minute - other people's messages will too.\n"
              "What matters is ORDER: check that no one else's booking is above your first copies.")
    else:
        for k in range(len(texts)):
            st = next((r[3] for r in rows if r[2] == keep[k]), None)
            res_txt = ("PASS" if st == tsec else f"stamped {st - tsec:+d} s from target") if st else "FAILED"
            print(f"RESULT '{_label(texts[k])}': {res_txt}")


async def snipe(client, a):
    peer = await client.get_input_entity(a.chat)
    hh, mm, ss = (list(map(int, a.at.split(":"))) + [0])[:3]
    today = dt.datetime.fromtimestamp(now(), IST).date()
    target = dt.datetime(today.year, today.month, today.day, hh, mm, ss, tzinfo=IST).timestamp()
    if target < now():
        sys.exit(f"Target {a.at} IST has already passed today.")
    print(f"Target arrival: {fmt(target)} IST  (+{a.margin_ms} ms margin)  chat={a.chat}")
    if a.burst > 1 and isinstance(peer, types.InputPeerChannel):
        try:
            full = await client(functions.channels.GetFullChannelRequest(peer))
            if getattr(full.full_chat, "slowmode_seconds", 0):
                print(f"  !! Group has SLOW MODE ({full.full_chat.slowmode_seconds}s) - only one message "
                      "allowed, so burst is switched OFF.")
                a.burst = 1
        except Exception:
            pass

    # Idle with keep-alive until 40 s before, then do fresh clock + latency measurements
    await wait_until(target - (50 if a.live_cal else 40), client)
    one_way = await prepare(client, ntp=not a.live_cal)
    live = None
    if a.live_cal:
        print(f"Live calibration against Telegram's clock (test sends to {a.calibrate_chat}, auto-deleted) ...")
        cpeer = await client.get_input_entity(a.calibrate_chat)
        ids = []
        try:
            q_lo, q_hi, ids = await measure_q(client, cpeer, a.text, 12)
            live = (q_lo, q_hi)
        except Exception as e:
            print(f"  live calibration failed ({type(e).__name__}) - falling back")
        finally:
            if ids:
                await client.delete_messages(a.calibrate_chat, ids)
    offset_ms, how = None, ""
    if live:
        offset_ms, how = -live[0] * 1000 + 10, "live calibration (+10 ms safety)"
    elif a.fire_offset_ms is not None:
        offset_ms, how = a.fire_offset_ms, "TG_FIRE_OFFSET_MS setting"
    elif os.path.isfile(CALIB_FILE) and not a.realtime:
        import json
        c = json.load(open(CALIB_FILE))
        age_h = (now() - c["measured_at"]) / 3600
        if age_h < 12:
            offset_ms, how = c["fire_offset_ms"], f"calibration.json ({age_h*60:.0f} min old)"
        else:
            print("  calibration.json is older than 12 h - ignoring it. Re-run --calibrate.")
    center_ms = None
    if live:
        center_ms = -live[0] * 1000           # earliest send that should not be early
    elif offset_ms is not None and how.startswith("calibration"):
        center_ms = c.get("edge_ms", offset_ms)       # burst centres on the measured edge
    elif offset_ms is not None:
        center_ms = offset_ms
    if offset_ms is not None:
        fire_at = target + offset_ms / 1000
        print(f"Using fire offset {offset_ms:+.0f} ms from {how}.")
    else:
        fire_at = target + a.margin_ms / 1000 - one_way
        if a.realtime:
            print(f"Aiming at REAL time {fmt(target)} + {a.margin_ms:.0f} ms (clock synced by NTP, "
                  f"one-way latency {one_way*1000:.0f} ms).")
        else:
            print("  !! No calibration found - guessing from latency. Run --calibrate first for accuracy.")
    print(f"Will fire at {fmt(fire_at)} IST. Do not touch anything.")

    if a.burst > 1 or len(a.texts) > 1:
        center = target + center_ms / 1000 if center_ms is not None else fire_at
        start = center - a.burst_early_ms / 1000
        print(f"Burst: {len(a.texts)} message(s) x {a.burst} copies from {fmt(start)} to "
              f"{fmt(start + a.burst_spread_ms / 1000)} (local clock)"
              + ("" if a.delete_extras else " - extras are NOT deleted"))
        await wait_until(start - 1.5, client)
        await client(_ping())
        return await burst_send(client, peer, a, target, start)

    await wait_until(fire_at - 1.5, client)
    await client(_ping())                    # last warm-up; nothing in flight at fire time
    first_req = send_req(peer, a.text)       # built in advance - zero work at fire time
    bytes(first_req)                         # warm up serializer
    await wait_until(fire_at)

    errors, attempt = [], 0
    deadline = fire_at + a.retry_for
    while True:
        attempt += 1
        t_send = now()
        try:
            res = await client(first_req if attempt == 1 else send_req(peer, a.text))
            t_done = now()
            break
        except FloodWaitError as e:
            print(f"Flood wait {e.seconds}s from Telegram - stopping.")
            return
        except RPCError as e:
            errors.append((fmt(t_send), type(e).__name__))
            if now() > deadline:
                print("Gave up retrying. Errors:", errors[-5:])
                return

    mid, date = extract(res)
    print("\n===== SENT =====")
    print(f"first attempt fired  : {fmt(fire_at)}")
    print(f"successful send at   : {fmt(t_send)}  (attempt #{attempt})")
    print(f"est. arrival at TG   : {fmt(t_send + one_way)}")
    print(f"confirmation at      : {fmt(t_done)}")
    if date:
        print(f"Telegram timestamp   : {date.astimezone(IST).strftime('%H:%M:%S')}  msg id {mid}")
        stamp, tsec = int(date.timestamp()), int(math.floor(target))
        if stamp == tsec:
            print(f"\nRESULT: PASS - Telegram stamped it {fmt(tsec)[:8]}, the target second.")
        elif stamp < tsec:
            print(f"\nRESULT: EARLY - stamped {stamp - tsec} s before target. Re-run --calibrate.")
        else:
            print(f"\nRESULT: LATE - stamped {stamp - tsec} s after target. Re-run --calibrate.")
    if errors:
        print(f"rejected attempts ({len(errors)}), first: {errors[0]}  -> chat was locked until it opened")


CODE_WHERE = {
    "SentCodeTypeApp": "inside the Telegram APP (chat named 'Telegram' with blue tick) - NOT by SMS",
    "SentCodeTypeSms": "by SMS",
    "SentCodeTypeCall": "by phone call",
    "SentCodeTypeFlashCall": "by a missed call (code = last digits of caller number)",
    "SentCodeTypeMissedCall": "by a missed call (code = last digits of caller number)",
    "SentCodeTypeEmailCode": "to your login email",
    "SentCodeTypeFragmentSms": "via Fragment (fragment.com)",
    "SentCodeTypeFirebaseSms": "by SMS",
    "SentCodeTypeSmsWord": "by SMS (a word instead of digits)",
    "SentCodeTypeSmsPhrase": "by SMS (a phrase instead of digits)",
    "SentCodeTypeSetUpEmailRequired": "nowhere yet - Telegram wants you to set up a login email "
                                      "in the official app first",
}


async def qr_login(client):
    """Log in by scanning a QR with the Telegram app - no code needed."""
    import webbrowser
    from telethon.errors import SessionPasswordNeededError
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        sys.exit("QR login needs one extra package:  pip install qrcode   (then run again)")
    qr = await client.qr_login()
    svg_path = os.path.abspath("login_qr.svg")
    while True:
        img = qrcode.make(qr.url, image_factory=qrcode.image.svg.SvgFillImage)
        img.save(svg_path)
        q = qrcode.QRCode(border=1)
        q.add_data(qr.url)
        q.print_ascii(invert=True)
        print(f"\nQR also saved as {svg_path} (opening it now).")
        print("On your PHONE: Telegram > Settings > Devices > Link Desktop Device > scan this QR.")
        webbrowser.open("file://" + svg_path)
        try:
            await qr.wait(timeout=max(5, qr.expires.timestamp() - time.time() - 1))
            break
        except asyncio.TimeoutError:
            print("\nQR expired - making a new one ...")
            await qr.recreate()
        except SessionPasswordNeededError:
            import getpass
            await client.sign_in(password=getpass.getpass("Two-step verification password: "))
            break
    try:
        os.remove(svg_path)
    except OSError:
        pass
    print("Logged in. Session saved; you won't need to scan again.")


async def login(client, phone):
    """Verbose login so you can see exactly where it gets stuck."""
    from telethon.errors import (SessionPasswordNeededError, PhoneNumberInvalidError,
                                 ApiIdInvalidError, PhoneCodeInvalidError, PhoneCodeExpiredError)
    print("Connecting to Telegram ...")
    import sqlite3
    try:
        await asyncio.wait_for(client.connect(), 20)
    except sqlite3.OperationalError:
        sys.exit("Session file is in use by another run of this script (another terminal, or a "
                 "stuck process). Close it first:  pkill -f tg_snipe.py   (or Ctrl+C in that terminal)")
    except (asyncio.TimeoutError, OSError) as e:
        sys.exit(f"Could not connect to Telegram ({type(e).__name__}). Your network/VPN/firewall "
                 "is probably blocking it - try a mobile hotspot.")
    if await client.is_user_authorized():
        return
    choice = input("\nLogin method:  [1] QR code - scan with Telegram on your phone (recommended)\n"
                   "               [2] code sent to the app / SMS\nChoose 1 or 2: ").strip()
    if choice != "2":
        return await qr_login(client)
    phone = phone or input("Phone number with country code (e.g. +9198xxxxxxxx): ").strip()
    phone = phone.replace(" ", "")
    try:
        sent = await client.send_code_request(phone)
    except ApiIdInvalidError:
        sys.exit("TG_API_ID / TG_API_HASH are wrong - copy them again from my.telegram.org.")
    except PhoneNumberInvalidError:
        sys.exit(f"Telegram says the phone number '{phone}' is invalid.")
    except FloodWaitError as e:
        sys.exit(f"Too many code requests. Telegram says wait {e.seconds // 60} min {e.seconds % 60} s.")

    while True:
        where = CODE_WHERE.get(type(sent.type).__name__, type(sent.type).__name__)
        print(f"\nLogin code sent {where}.")
        if sent.next_type:
            print("  (If it doesn't arrive, press Enter without typing to resend via "
                  f"{CODE_WHERE.get(type(sent.next_type).__name__, type(sent.next_type).__name__)})")
        code = input("Enter code: ").strip()
        if code:
            break
        if not sent.next_type:
            print("  No other delivery method available - check the Telegram app on all your devices.")
            continue
        try:
            sent = await client(functions.auth.ResendCodeRequest(phone, sent.phone_code_hash))
        except FloodWaitError as e:
            print(f"  Resend blocked, wait {e.seconds} s.")

    try:
        await client.sign_in(phone, code, phone_code_hash=sent.phone_code_hash)
    except SessionPasswordNeededError:
        import getpass
        await client.sign_in(password=getpass.getpass("Two-step verification password: "))
    except (PhoneCodeInvalidError, PhoneCodeExpiredError) as e:
        sys.exit(f"{type(e).__name__}: wrong or expired code - run again.")
    print("Logged in. Session saved; you won't need the code again.")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-id", default=cfg("TG_API_ID"))
    p.add_argument("--api-hash", default=cfg("TG_API_HASH"))
    p.add_argument("--phone", default=cfg("TG_PHONE"))
    p.add_argument("--session", default=cfg("TG_SESSION") or "snipe")
    p.add_argument("--calibrate", action="store_true", help="test timing using Saved Messages")
    p.add_argument("--test", action="store_true",
                   help="full rehearsal: real-run flow, but to Saved Messages at the next minute")
    p.add_argument("--list-chats", nargs="?", const="", metavar="FILTER",
                   help="print your chats with the value to use for TG_CHAT, optionally filtered by name")
    p.add_argument("--chat", default=cfg("TG_CHAT"), help="@username, t.me link, or chat id")
    p.add_argument("--text", default=cfg("TG_TEXT"), help="message to send")
    p.add_argument("--at", default=cfg("TG_AT") or "10:03:00", help="IST time HH:MM[:SS]")
    p.add_argument("--margin-ms", type=float, default=float(cfg("TG_MARGIN_MS") or 10),
                   help="land this many ms AFTER the target (avoid arriving at 10:02:59.999)")
    p.add_argument("--retry-for", type=float, default=float(cfg("TG_RETRY_FOR") or 3),
                   help="keep re-sending for N s if the chat is still locked")
    p.add_argument("--fire-offset-ms", type=float,
                   default=float(cfg("TG_FIRE_OFFSET_MS")) if cfg("TG_FIRE_OFFSET_MS") else None,
                   help="press send this many ms relative to target (overrides calibration.json)")
    p.add_argument("--calibrate-chat", default=cfg("TG_CALIBRATE_CHAT") or "me",
                   help="chat used for calibration (default: your Saved Messages)")
    p.add_argument("--burst", type=int, default=int(cfg("TG_BURST") or 1),
                   help="send N copies around the target, keep only the first on-time one")
    p.add_argument("--burst-spread-ms", type=float, default=float(cfg("TG_BURST_SPREAD_MS") or 300),
                   help="time window the burst copies are spread over")
    p.add_argument("--burst-early-ms", type=float, default=float(cfg("TG_BURST_EARLY_MS") or 0),
                   help="start the burst this many ms before the estimated edge")
    p.add_argument("--delete-extras", type=int, default=int(cfg("TG_DELETE_EXTRAS") or 0),
                   help="1 = delete early/duplicate copies, 0 = keep all")
    p.add_argument("--live-cal", type=int, default=int(cfg("TG_LIVE_CALIBRATION") or 1),
                   help="1 = calibrate against Telegram's clock ~1 min before target (recommended)")
    p.add_argument("--reference", default=cfg("TG_REFERENCE") or "realtime",
                   choices=["realtime", "telegram"],
                   help="realtime = aim at real 10:03:00 (what phones show, via NTP); "
                        "telegram = aim at the group's Telegram timestamp")
    p.add_argument("--text-file", default=cfg("TG_TEXT_FILE"), help="file containing the message")
    a = p.parse_args()
    a.realtime = a.reference == "realtime"
    if a.realtime:
        a.live_cal = 0

    # Message: file (exact, multi-line) > TG_TEXT (\n = new line)
    # TG_TEXT_FILE may list several files separated by commas -> several messages
    a.texts = []
    here = os.path.dirname(os.path.abspath(__file__))
    for tf in [x.strip() for x in (a.text_file or "").split(",") if x.strip()]:
        if not os.path.isabs(tf) and not os.path.isfile(tf):
            tf = os.path.join(here, tf)
        if os.path.isfile(tf):
            with open(tf, encoding="utf-8") as f:
                t = f.read().rstrip("\r\n")
            if t.strip():
                a.texts.append(t)
        elif a.list_chats is None:
            print(f"  !! message file not found: {tf}")
    if not a.texts and a.text:
        a.texts = [a.text.replace("\\n", "\n")]
    a.text = a.texts[0] if a.texts else None
    if a.texts and a.list_chats is None:
        for i, t in enumerate(a.texts, 1):
            print(f"Message {i} of {len(a.texts)}:\n----------------\n{t}\n----------------"
                  f"\n({len(t)} chars)\n")

    missing = [n for n, v in (("TG_API_ID", a.api_id), ("TG_API_HASH", a.api_hash)) if not v]
    if a.test:
        a.chat = a.calibrate_chat if a.calibrate_chat != "me" else "me"
        if "--at" not in sys.argv:          # next whole minute at least 110 s away (room for live calibration)
            t = dt.datetime.fromtimestamp(time.time() + 110, IST).replace(second=0, microsecond=0)
            t += dt.timedelta(minutes=1)
            a.at = t.strftime("%H:%M:%S")
        print(f"TEST MODE: sending to {a.chat} (not the real group) at {a.at} IST\n")
    if not a.calibrate and a.list_chats is None:
        missing += [n for n, v in (("TG_CHAT", a.chat), ("TG_TEXT", a.text)) if not v]
    if missing:
        sys.exit("Please fill in: " + ", ".join(missing) + "  (in .env or the CONFIG block)")
    chat = a.chat
    if chat and chat.lstrip("-").isdigit():
        chat = int(chat)                     # numeric chat ids must be int for Telethon
    a.chat = chat
    if str(a.calibrate_chat).lstrip("-").isdigit():
        a.calibrate_chat = int(a.calibrate_chat)

    client = TelegramClient(a.session, int(a.api_id), a.api_hash)
    client.flood_sleep_threshold = 0
    await login(client, a.phone)
    me = await client.get_me()
    print(f"Logged in as {me.first_name} (DC {client.session.dc_id})")

    if a.list_chats is not None:
        print(f"\n{'TG_CHAT value':<26} {'type':<8} name")
        async for d in client.iter_dialogs():
            if a.list_chats.lower() not in (d.name or "").lower():
                continue
            kind = "channel" if d.is_channel and not d.is_group else "group" if d.is_group else "user/bot"
            user = getattr(d.entity, "username", None)
            print(f"{('@' + user) if user else str(d.id):<26} {kind:<8} {d.name}")
    elif a.calibrate:
        await calibrate(client, a.text, a.calibrate_chat)
    else:
        if not a.chat or not a.text:
            sys.exit("--chat and --text are required (or use --calibrate)")
        await snipe(client, a)
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())