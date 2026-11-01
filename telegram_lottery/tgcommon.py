"""Shared helpers: .env loading, client creation, QR login."""
import asyncio
import os
import sys

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path=None):
    path = path or os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def make_client():
    load_env()
    api_id, api_hash = os.environ.get("TG_API_ID"), os.environ.get("TG_API_HASH")
    if not api_id or not api_hash:
        sys.exit("Set TG_API_ID and TG_API_HASH in .env (see .env.example)")
    return TelegramClient(os.path.join(HERE, "snipe"), int(api_id), api_hash)


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
