#!/usr/bin/env python3
"""
snipe.py - make ONE Telegram message ARRIVE at Telegram's server at an exact instant,
measured on TELEGRAM'S OWN CLOCK (see telegram_clock.py for the method).

  1. While waiting, ping Telegram continuously (1 Hz -> 5 Hz -> 20 Hz as the target nears).
     Each ping is stamped by the server when handled, which gives S = (server time at
     arrival) - (our send time) for exactly the path the real message will take.
  2. Fit S's drift, pick a low quantile of its jitter (so we are almost never EARLY), and
     compute  fire_time = target - S.
  3. Stop pinging 0.5 s before, disable GC, sleep to ~10 ms before, then busy-wait on the
     monotonic clock and push the pre-built, pre-serialised request into Telethon's send
     queue at that exact nanosecond.

--dry-run does all of this but fires a PING instead of the message, and reports where the
server stamped it relative to the target. Use it to see your real-world precision.

Usage:
    python snipe.py --at 10:51:00 --dry-run          # chat from TG_CHAT, text from message.txt
    python snipe.py --at 10:51:00.000500
    python snipe.py --at 10:51:00 --burst 3 --burst-gap-us 300   # hedge against jitter
First run shows a QR code: Telegram app -> Settings -> Devices -> Link Desktop Device.
"""
import argparse
import asyncio
import datetime as dt
import gc
import os
import random
import sys
import time

from telethon import functions
from telethon.errors import UsernameInvalidError
from telethon.network.requeststate import RequestState

from telegram_clock import TelegramClock
from tgcommon import HERE, load_env, make_client, qr_login


def parse_target(text):
    """'HH:MM:SS[.ffffff]' (local, today or next occurrence) or full ISO datetime -> unix ns."""
    if "-" in text:
        d = dt.datetime.fromisoformat(text)
    else:
        t = dt.datetime.strptime(text, "%H:%M:%S.%f" if "." in text else "%H:%M:%S").time()
        d = dt.datetime.combine(dt.date.today(), t)
    d = d.astimezone() if d.tzinfo is None else d
    if "-" not in text and d.timestamp() < time.time():
        d += dt.timedelta(days=1)
    return int(d.timestamp()) * 10**9 + d.microsecond * 1000


def fmt(ns):
    d = dt.datetime.fromtimestamp(ns // 10**9).replace(microsecond=(ns // 1000) % 10**6)
    return f"{d:%Y-%m-%d %H:%M:%S.%f}"


async def resolve_chat(client, chat):
    chat = chat.strip()
    if chat.lstrip("-").isdigit():
        chat = int(chat)
    try:
        return await client.get_input_entity(chat)
    except ValueError:                                   # id not in session cache yet
        await client.get_dialogs()
        try:
            return await client.get_input_entity(chat)
        except ValueError:
            sys.exit(f"Chat {chat!r} not found among your chats. Join it first, or use @username / invite link.")
    except UsernameInvalidError:
        sys.exit(f"No Telegram user/chat with username {chat!r}.")


def cadence(eta_s):
    return 1.0 if eta_s > 120 else 0.2 if eta_s > 15 else 0.05


def message_date(res):
    for u in getattr(res, "updates", None) or ():
        m = getattr(u, "message", None)
        if m is not None and getattr(m, "date", None):
            return m.date
    return getattr(res, "date", None)


async def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", default=os.environ.get("TG_CHAT"),
                    help="@username, invite link or numeric id (default: TG_CHAT from .env)")
    ap.add_argument("--text", help="message text (default: contents of --text-file)")
    ap.add_argument("--text-file", default=os.path.join(HERE, "message.txt"))
    ap.add_argument("--at", required=True, help="Telegram-server time: local HH:MM:SS[.ffffff] or ISO datetime")
    ap.add_argument("--quantile", type=float, default=0.1,
                    help="chance of landing EARLY of the target (default 0.1; lower = safer, later)")
    ap.add_argument("--lead-ms", type=float, default=0.0, help="manual extra early-fire (+) or late (-), ms")
    ap.add_argument("--burst", type=int, default=1, help="send N copies (default 1)")
    ap.add_argument("--burst-gap-us", type=float, default=200, help="gap between burst copies, microseconds")
    ap.add_argument("--dry-run", action="store_true", help="fire a ping instead of the message, report precision")
    args = ap.parse_args()

    if args.text is None:
        if not os.path.exists(args.text_file):
            sys.exit(f"Pass --text or create {args.text_file}")
        args.text = open(args.text_file, encoding="utf-8").read().strip()
    print(f"Message:\n---\n{args.text}\n---")
    if not args.chat:
        sys.exit("Pass --chat or set TG_CHAT in .env")

    client = make_client()
    await client.connect()
    tg = TelegramClock(client)
    try:
        await qr_login(client)
        peer = await resolve_chat(client, args.chat)

        target_ns = parse_target(args.at)
        print("Calibrating against Telegram's clock...")
        await tg.start(warmup=20)
        eta = (target_ns - tg.now_ns()) / 1e9
        print(f"Target {fmt(target_ns)} (Telegram time), {eta:.1f}s from now. "
              f"Telegram clock vs local: {(tg.now_ns() - time.time_ns()) / 1e6:+.3f} ms")
        if eta < 3:
            sys.exit("Target too soon; need at least ~3 s.")

        # Pre-build every request now (serialisation cost stays out of the critical path).
        pings = [tg.build_ping() for _ in range(args.burst)]
        msgs = [RequestState(functions.messages.SendMessageRequest(
                    peer=peer, message=args.text, random_id=random.getrandbits(63)))
                for _ in range(args.burst)]

        last_print = 0
        while True:
            eta = (target_ns - tg.now_ns()) / 1e9
            if eta < 0.5:
                break
            tg.ping_every_s = cadence(eta)
            if time.time() - last_print > 5:
                last_print = time.time()
                try:
                    s, info = tg.predict_S(time.perf_counter_ns(), args.quantile)
                    print(f"  T-{eta:8.1f}s  samples {info['n']:4d}  RTT min {info['rtt_min_ms']:.1f} ms  "
                          f"drift {info['slope_ppm']:+.1f} ppm  jitter p10/p50/p90 "
                          f"{info['jitter_p10_us']:.0f}/{info['jitter_p50_us']:.0f}/{info['jitter_p90_us']:.0f} us")
                except RuntimeError:
                    pass
            await asyncio.sleep(0.05)

        tg.paused = True                                  # quiet line for the final shot
        await asyncio.sleep(0.15)                         # let in-flight pongs land
        eta_ns = target_ns - tg.now_ns()
        s_hat, info = tg.predict_S(time.perf_counter_ns() + eta_ns, args.quantile)
        fire_perf = target_ns - int(s_hat) - int(args.lead_ms * 1e6)
        print(f"Final: {info['n']} samples, RTT min {info['rtt_min_ms']:.2f} ms, drift {info['slope_ppm']:+.1f} ppm, "
              f"jitter p10/p50/p90/p99 {info['jitter_p10_us']:.0f}/{info['jitter_p50_us']:.0f}/"
              f"{info['jitter_p90_us']:.0f}/{info['jitter_p99_us']:.0f} us")
        print(f"Firing in {(fire_perf - time.perf_counter_ns()) / 1e6:.1f} ms "
              f"(target - S, S = {s_hat / 1e6:+.3f} ms rel. to local monotonic clock)")

        gc.collect()
        gc.disable()
        slack = fire_perf - time.perf_counter_ns() - 10_000_000
        if slack < 0:
            sys.exit("Missed the window while preparing; nothing sent.")
        await asyncio.sleep(slack / 1e9)

        sent = []                                         # (deadline, actual_send_perf, future, ping_id)
        for i in range(args.burst):
            dl = fire_perf + int(i * args.burst_gap_us * 1000)
            if i == 0:
                while time.perf_counter_ns() < dl:        # tight spin, ~1 us resolution
                    pass
            else:
                while time.perf_counter_ns() < dl:        # cooperative: lets the previous send flush
                    await asyncio.sleep(0)
            t = time.perf_counter_ns()
            if args.dry_run:
                fut = tg.enqueue(pings[i][1])
            else:
                fut = tg.enqueue(msgs[i])
            sent.append((dl, t, fut, pings[i][0]))
            await asyncio.sleep(0)
        gc.enable()

        for i, (dl, t, fut, pid) in enumerate(sent):
            try:
                res = await asyncio.wait_for(fut, 10)
            except Exception as e:
                print(f"  copy {i + 1}: ERROR {e!r}")
                continue
            line = f"  copy {i + 1}: trigger error {(t - dl) / 1e3:+.1f} us"
            if args.dry_run:
                pong = tg._pong_ns.pop(pid, None)
                if pong is not None:
                    err = (pong - (target_ns + i * int(args.burst_gap_us * 1000))) / 1e3
                    line += f", server stamped ping at target {err:+.0f} us  (<0 early, >0 late)"
            else:
                d = message_date(res)
                line += f", accepted (server message date {d:%H:%M:%S})" if d else ", accepted"
            print(line)
    finally:
        tg.stop()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
