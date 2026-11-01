#!/usr/bin/env python3
"""
telegram_clock.py - Telegram's SERVER clock, measured over the live MTProto connection.

Every packet from Telegram carries a msg_id: top 32 bits = server unix seconds, bottom 32
bits = fraction of the second. We timestamp each packet on arrival (monotonic clock) and use
two measurements:

 * DOWNSTREAM  server_stamp - local_arrival = offset - delay_down (delay >= 0), so the
   maximum over a window is the least-delayed sample. +RTT/2 gives the clock for display.
 * UPSTREAM    for a ping we send at local time t0, the server stamps its pong at the
   moment it handled the ping:   S = pong_stamp - t0 = offset + delay_up (+ tiny server work).
   A message sent through the SAME code path at local time t therefore reaches the server
   at server-time  t + S.  No symmetric-path assumption, and constant overheads (event loop
   hops, encryption, kernel) cancel because pings travel exactly like the real message.
   S drifts (your oscillator vs. Telegram's), so a line is fitted to the lower envelope.

Usage:
    python telegram_clock.py              # live display, Ctrl+C to stop
    python telegram_clock.py --utc
Reusable: tc = TelegramClock(client); await tc.start(); tc.now_ns(); tc.predict_S(...)
"""
import argparse
import asyncio
import collections
import datetime as dt
import random
import sys
import time

from telethon import functions
from telethon.network.requeststate import RequestState

from tgcommon import make_client, qr_login

WINDOW_S = 30          # samples used for the S estimate
KEEP_S = 60
BUCKET_NS = 2 * 10**9  # lower-envelope bucket for the drift fit
MAX_DRIFT = 200e-6     # clamp fitted drift to 200 ppm


def msg_id_to_ns(msg_id):
    return (msg_id >> 32) * 10**9 + (((msg_id & 0xFFFFFFFF) * 10**9) >> 32)


def _objs(msg):
    yield msg.msg_id, msg.obj
    for inner in getattr(msg.obj, "messages", None) or ():
        yield from _objs(inner)


def quantile(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))]


class TelegramClock:
    def __init__(self, client):
        self.client = client
        self._raw = collections.deque()      # (arrival_perf, server_ns - arrival_perf)
        self._up = collections.deque()       # (send_perf, S, rtt)
        self._pong_ns = {}                   # ping_id -> server stamp of the pong
        self.ping_every_s = 0.05
        self.paused = False
        self._task = None

    # -- low-level plumbing --------------------------------------------------
    @property
    def _sender(self):
        return self.client._sender

    def _hook(self):
        state = self._sender._state
        if getattr(state, "_tgclock_hooked", False):
            return
        orig = state.decrypt_message_data

        def wrapped(body):
            arrival = time.perf_counter_ns()      # stamped before any decrypt work
            msg = orig(body)
            try:
                for mid, obj in _objs(msg):
                    ns = msg_id_to_ns(mid)
                    self._raw.append((arrival, ns - arrival))
                    if type(obj).__name__ == "Pong":
                        self._pong_ns[obj.ping_id] = ns
            except Exception:
                pass                              # never break Telethon's receive loop
            return msg

        state.decrypt_message_data = wrapped
        state._tgclock_hooked = True

    def enqueue(self, state):
        """Hand a pre-built RequestState to Telethon's send loop; returns its future."""
        self._sender._send_queue.append(state)
        return state.future

    def build_ping(self):
        pid = random.getrandbits(62)
        return pid, RequestState(functions.PingRequest(ping_id=pid))

    def _trim(self, now):
        cutoff = now - KEEP_S * 10**9
        for dq in (self._raw, self._up):
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    async def ping_once(self):
        pid, state = self.build_ping()
        t0 = time.perf_counter_ns()
        fut = self.enqueue(state)
        await fut
        t1 = time.perf_counter_ns()
        ns = self._pong_ns.pop(pid, None)
        if ns is not None:
            self._up.append((t0, ns - t0, t1 - t0))
            self._trim(t1)
        return t0, t1, ns

    async def _ping_loop(self):
        while True:
            if not self.paused:
                self._hook()                      # re-arm if the sender was rebuilt
                try:
                    await self.ping_once()
                except Exception:
                    await asyncio.sleep(1)
                    continue
            await asyncio.sleep(self.ping_every_s)

    async def start(self, warmup=20):
        self._hook()
        self._task = asyncio.ensure_future(self._ping_loop())
        while len(self._up) < warmup:
            await asyncio.sleep(0.05)

    def stop(self):
        if self._task:
            self._task.cancel()

    # -- readouts ------------------------------------------------------------
    @property
    def min_rtt_ns(self):
        return min(r for _, _, r in self._up)

    def now_ns(self):
        """Telegram server unix time in ns (downstream-max + RTT/2 estimate)."""
        off = max(o for _, o in self._raw) + self.min_rtt_ns // 2
        return time.perf_counter_ns() + off

    def predict_S(self, at_perf_ns, q=0.1):
        """
        Predict S (server_time_at_arrival - send_perf) for a send at `at_perf_ns`.
        q = chance the message lands EARLIER than the target (0.1 -> almost never early).
        Returns (S_hat_ns, info).
        """
        pts = [(t, s) for t, s, _ in self._up if t > at_perf_ns - WINDOW_S * 10**9]
        if len(pts) < 5:
            raise RuntimeError("not enough ping samples yet")
        # lower envelope: min S in each bucket
        t_first = pts[0][0]
        mins = {}
        for t, s in pts:
            k = (t - t_first) // BUCKET_NS
            if k not in mins or s < mins[k][1]:
                mins[k] = (t, s)
        env = list(mins.values())
        slope = 0.0                                # ns per second
        if len(env) >= 4 and env[-1][0] - env[0][0] >= 6 * 10**9:
            xs = [(t - at_perf_ns) / 1e9 for t, _ in env]
            ys = [s for _, s in env]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            if abs(b) <= MAX_DRIFT * 1e9:
                slope = b
        if slope:
            a = my - slope * mx
        else:
            a = min(s for _, s in pts)
        res = sorted(s - (a + slope * (t - at_perf_ns) / 1e9) for t, s in pts)
        s_hat = a + quantile(res, q)
        info = {"n": len(pts), "slope_ppm": slope / 1e3,
                "rtt_min_ms": min(r for _, _, r in self._up) / 1e6,
                "jitter_p10_us": quantile(res, 0.1) / 1e3, "jitter_p50_us": quantile(res, 0.5) / 1e3,
                "jitter_p90_us": quantile(res, 0.9) / 1e3, "jitter_p99_us": quantile(res, 0.99) / 1e3}
        return s_hat, info


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utc", action="store_true", help="show UTC instead of local time")
    ap.add_argument("--hz", type=float, default=50, help="screen refresh rate (default 50)")
    args = ap.parse_args()

    client = make_client()
    await client.connect()
    clock = TelegramClock(client)
    try:
        await qr_login(client)
        print("Syncing to Telegram server clock...")
        await clock.start()
        clock.ping_every_s = 0.25
        tz = dt.timezone.utc if args.utc else None
        print("Ctrl+C to stop.\n")
        while True:
            tg = clock.now_ns()
            d = dt.datetime.fromtimestamp(tg // 10**9, tz).replace(microsecond=(tg // 1000) % 10**6)
            diff_ms = (tg - time.time_ns()) / 1e6
            sys.stdout.write(
                f"\rTelegram {d:%Y-%m-%d %H:%M:%S.%f}   local clock is {-diff_ms:+9.3f} ms vs TG   "
                f"RTT min {clock.min_rtt_ns / 1e6:5.1f} ms (+-{clock.min_rtt_ns / 2e6:4.1f})   "
                f"samples {len(clock._up):4d}  ")
            sys.stdout.flush()
            await asyncio.sleep(1 / args.hz)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        clock.stop()
        await client.disconnect()
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
