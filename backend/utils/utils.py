import uuid
from typing import List


def generate_id() -> str:
    raw = uuid.uuid4().bytes
    encoded = _base62_encode(raw)
    return f"{encoded[:12]}"


def _base62_encode(data: bytes) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    num = int.from_bytes(data, "big")
    if num == 0:
        return alphabet[0]
    chars: List[str] = []
    while num:
        num, rem = divmod(num, 62)
        chars.append(alphabet[rem])
    return "".join(reversed(chars))
