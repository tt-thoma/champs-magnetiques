import unittest
import uuid
import warnings
from io import BufferedReader, BufferedWriter
from pathlib import Path

TAG_FILE: Path = Path("./.numba_cache/tag")


class TestCache(unittest.TestCase):
    def test_cache(self) -> None:
        """Display numba_cache tag and reset it"""
        if TAG_FILE.exists():
            file: BufferedReader
            with open(TAG_FILE, "rb") as file:
                print(uuid.UUID(bytes=file.read()))
        else:
            warnings.warn("Cache unavailable")

        new_cache: uuid.UUID = uuid.uuid4()
        wfile: BufferedWriter
        with open(TAG_FILE, "wb") as wfile:
            wfile.write(new_cache.bytes)
        print(new_cache)


if __name__ == "__main__":
    unittest.main()
