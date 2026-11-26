#!/usr/bin/env python3
"""Downloads the USC-SIPI "Miscellaneous" volume and converts it to PGM.

Source: https://sipi.usc.edu/database/database.php?volume=misc
The archive holds 39 TIFF images (256x256 to 1024x1024, gray and colour).
Colour images are converted to 8-bit grayscale. Requires Pillow.

Usage:
  python3 scripts/fetch_sipi.py --out data/input
"""

import argparse
import io
import os
import sys
import tarfile
import urllib.request

URLS = [
    "https://sipi.usc.edu/database/misc.tar.gz",
    "http://sipi.usc.edu/database/misc.tar.gz",
]


def download():
    for url in URLS:
        try:
            print("downloading", url)
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError as err:  # URLError and socket errors.
            print("  failed:", err)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/input")
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is required: python3 -m pip install --user pillow")

    blob = download()
    if blob is None:
        sys.exit("could not download the SIPI archive; use "
                 "scripts/generate_synthetic.py instead")

    os.makedirs(args.out, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = os.path.basename(member.name)
            if not member.isfile() or not name.lower().endswith(
                    (".tif", ".tiff")):
                continue
            data = archive.extractfile(member).read()
            image = Image.open(io.BytesIO(data)).convert("L")
            stem = os.path.splitext(name)[0].replace(".", "_")
            path = os.path.join(args.out, "sipi_%s.pgm" % stem)
            image.save(path)
            count += 1
            print("wrote %s (%dx%d)" % (path, image.width, image.height))
    print("converted %d SIPI images into %s" % (count, args.out))


if __name__ == "__main__":
    main()
