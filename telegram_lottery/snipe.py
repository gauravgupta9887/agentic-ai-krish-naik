#!/usr/bin/env python3
"""
snipe.py - fire one Telegram message so it ARRIVES at the server at a chosen instant.

How it gets as close as the network allows:
  1. SNTP-syncs to atomic time (your PC clock is typically 20-500 ms off).
  2. Pings Telegram over the live MTProto connection and measures one-way latency (RTT/2).
  3. Pre-connects, pre-resolves the chat and pre-builds the request, then sleeps until
     ~20 ms before the fire time and busy-waits on a monotonic clock for the last bit.
  4. Fires at (target - one_way_latency) so the packet lands at the target.

Honest limit: internet jitter is ~1-10 ms, so "microsecond" is the precision of the
local trigger, not a guarantee of arrival. Use --lead-ms to nudge by hand.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # fill in api_id / api_hash
    python snipe.py --chat @mygroup --text "hello" --at 10:51:00
    python snipe.py --chat @mygroup --text "hello" --at 10:51:00.000500 --dry-run
First run shows a QR code: Telegram app -> Settings -> Devices -> Link Desktop Device.
"""
import argparse
import asyncio
import datetime as dt
import os
import random
import socket
import statistics
import struct
import sys
import time

from telethon import TelegramClient, functions
from telethon.errors import SessionPasswordNeededError

HERE = os.path.dirname(os.path.abspath(__file__))
NTP_SERVERS = ["time.google.com", "time.cloudflare.com", "pool.ntp.org", "time.windows.com"]
NTP_EPOCH_DELTA = 2208988800  # seconds between 1900 and 1970


def load_env(path):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


# ---------------------------------------------------------------- clock ----
def sntp_offset_ns(server, timeout=1.5):
    """Return (offset_ns, rtt_ns) of local clock vs server; offset = server - local."""
    pkt = b"\x1b" + 47 * b"\0"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        t0 = time.time_ns()
        s.sendto(pkt, (server, 123))
        data, _ = s.recvfrom(512)
        t3 = time.time_ns()
    f = struct.unpack("!12I", data[:48])

    def to_ns(sec, frac):
        return (sec - NTP_EPOCH_DELTA) * 10**9 + (frac * 10**9 >> 32)

    t1, t2 = to_ns(f[8], f[9]), to_ns(f[10], f[11])
    return ((t1 - t0) + (t2 - t3)) // 2, (t3 - t0) - (t2 - t1)


class Clock:
    """True wall-clock ns = anchored local time advanced by a monotonic counter."""

    def __init__(self):
        samples = []
        for srv in NTP_SERVERS:
            for _ in range(4):
                try:
                    samples.append(sntp_offset_ns(srv))
                except OSError:
                    break
        if not samples:
            print("! NTP unreachable, falling back to local clock (may be off by 100s of ms)")
            offset = 0
        else:
            samples.sort(key=lambda x: x[1])            # lowest RTT = most trustworthy
            best = samples[: max(3, len(samples) // 4)]
            offset = int(statistics.median(o for o, _ in best))
            print(f"NTP offset {offset / 1e6:+.3f} ms  (best RTT {samples[0][1] / 1e6:.1f} ms, "
                  f"{len(samples)} samples)")
        self._wall0 = time.time_ns() + offset
        self._perf0 = time.perf_counter_ns()

    def now_ns(self):
        return self._wall0 + (time.perf_counter_ns() - self._perf0)

    def perf_for(self, wall_ns):
        return self._perf0 + (wall_ns - self._wall0)


def parse_target(text):
    """'HH:MM:SS[.ffffff]' (local, today or next occurrence) or full ISO datetime."""
    if "-" in text:
        d = dt.datetime.fromisoformat(text)
    else:
        t = dt.datetime.strptime(text, "%H:%M:%S.%f" if "." in text else "%H:%M:%S").time()
        d = dt.datetime.combine(dt.date.today(), t)
    d = d.astimezone() if d.tzinfo is None else d
    if "-" not in text and d.timestamp() < time.time():
        d += dt.timedelta(days=1)
    return int(d.timestamp()) * 10**9 + d.microsecond * 1000


# ------------------------------------------------------------- telegram ----
async def qr_login(client):
    if await client.is_user_authorized():
        return
    import qrcode
    qr = await client.qr_login()
    while True:
        q = qrcode.QRCode(border=1)
        q.add_data(qr.url)
        q.make()
        print("\nScan: Telegram app -> Settings -> Devices -> Link Desktop Device\n")
        q.print_ascii(invert=True)
        try:
            await qr.wait(timeout=25)
            break
        except asyncio.TimeoutError:
            await qr.recreate()                          # token expires ~30 s
        except SessionPasswordNeededError:
            await client.sign_in(password=input("2FA password: "))
            break
    print("Logged in.")


async def one_way_latency_ns(client, n=15):
    rtts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        await client(functions.PingRequest(ping_id=random.getrandbits(62)))
        rtts.append(time.perf_counter_ns() - t0)
        await asyncio.sleep(0.05)
    rtts.sort()
    low = rtts[: max(3, n // 3)]                         # drop the jittery tail
    print(f"Ping RTT min {rtts[0] / 1e6:.1f} ms, median {statistics.median(rtts) / 1e6:.1f} ms")
    return int(statistics.median(low)) // 2


async def main():
    load_env(os.path.join(HERE, ".env"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", required=True, help="@username, invite link, phone or numeric id")
    ap.add_argument("--text", required=True)
    ap.add_argument("--at", required=True, help="local HH:MM:SS[.ffffff] or ISO datetime")
    ap.add_argument("--lead-ms", type=float, default=0.0, help="extra manual early-fire (+) or late (-) in ms")
    ap.add_argument("--burst", type=int, default=1, help="send N copies, 1 ms apart (default 1)")
    ap.add_argument("--dry-run", action="store_true", help="do everything but don't send")
    args = ap.parse_args()

    api_id, api_hash = os.environ.get("TG_API_ID"), os.environ.get("TG_API_HASH")
    if not api_id or not api_hash:
        sys.exit("Set TG_API_ID and TG_API_HASH in .env (see .env.example)")

    clock = Clock()
    target_ns = parse_target(args.at)
    secs_left = (target_ns - clock.now_ns()) / 1e9
    print(f"Target {dt.datetime.fromtimestamp(target_ns / 1e9):%Y-%m-%d %H:%M:%S.%f}  ({secs_left:.1f}s from now)")
    if secs_left < 5:
        sys.exit("Target too soon; need at least ~5 s to connect and calibrate.")

    client = TelegramClient(os.path.join(HERE, "snipe"), int(api_id), api_hash)
    await client.connect()
    try:
        await qr_login(client)
        peer = await client.get_input_entity(args.chat)  # resolved ahead of time
        latency_ns = await one_way_latency_ns(client)
        fire_ns = target_ns - latency_ns - int(args.lead_ms * 1e6)
        print(f"One-way latency ~{latency_ns / 1e6:.1f} ms -> firing at target - "
              f"{(target_ns - fire_ns) / 1e6:.1f} ms")

        reqs = [functions.messages.SendMessageRequest(peer=peer, message=args.text,
                                                      random_id=random.getrandbits(63))
                for _ in range(args.burst)]

        # Coarse sleep, keeping the link warm with a ping, then busy-wait the last 20 ms.
        while (fire_ns - clock.now_ns()) > 2 * 10**9:
            await asyncio.sleep(1)
            await client(functions.PingRequest(ping_id=random.getrandbits(62)))
        await asyncio.sleep(max(0, (fire_ns - clock.now_ns() - 20_000_000) / 1e9))
        deadline = clock.perf_for(fire_ns)
        while time.perf_counter_ns() < deadline:
            pass
        sent_ns = clock.now_ns()
        tasks = []
        for r in reqs:
            if not args.dry_run:
                tasks.append(asyncio.ensure_future(client(r)))
            await asyncio.sleep(0.001 if len(reqs) > 1 else 0)
        err = (sent_ns - fire_ns) / 1e3
        print(f"Fired at {dt.datetime.fromtimestamp(sent_ns / 1e9):%H:%M:%S.%f} "
              f"(trigger error {err:+.0f} us){'  [dry run]' if args.dry_run else ''}")
        if tasks:
            res = await asyncio.gather(*tasks, return_exceptions=True)
            for x in res:
                print("  ->", "ERROR " + repr(x) if isinstance(x, Exception) else "accepted by Telegram")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
