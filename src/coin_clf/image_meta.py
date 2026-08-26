"""Shared image-metadata extraction -- width/height/mode, one definition.

Pulled out of coin_clf.validate_batch._build_frame so the retraining gate and the serving
layer's prediction log read metadata through the SAME code, rather than each carrying its own
idea of what "this image's mode" means. Two implementations agree right up until they quietly
don't, which is the shape of the drift problems this project already paid for once.

stdlib + PIL ONLY, deliberately. validate_batch imports pandas at module scope and
requirements-serve.txt has no pandas, so app/main.py cannot import validate_batch without
breaking the serving container at startup. This module is the part both sides can share:
the gate keeps its pandas frame, serving keeps its slim image, and the extraction is one thing.

Two entry points, one definition of the metadata itself:
    image_metadata(img)      -- from an already-open PIL Image (serving: bytes in memory)
    metadata_from_path(path) -- open, verify, re-open (the gate: files on disk)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Everything PIL throws for a file that isn't a decodable image. Lives here, next to the only
# code that opens images, so callers can't drift on what "unreadable" means either.
UNREADABLE_EXCEPTIONS = (OSError, SyntaxError, UnidentifiedImageError)


@dataclass(frozen=True)
class ImageMeta:
    """The three metadata fields both the gate and the prediction log care about."""

    width: int
    height: int
    mode: str


def image_metadata(img: Image.Image) -> ImageMeta:
    """Metadata of an open PIL Image, AS OPENED.

    Callers that convert must call this BEFORE converting. app/main.py's /predict does
    .convert("RGB") on every upload before preprocessing; metadata read after that convert
    reports the CONVERTED mode, so every logged row would say "RGB" -- including a batch of
    deliberately grayscaled images. That makes mode drift permanently unobservable, which
    defeats the point of logging mode at all.
    """
    width, height = img.size
    return ImageMeta(width=width, height=height, mode=img.mode)


def metadata_from_path(path: str | Path) -> ImageMeta | None:
    """Metadata of an image FILE, or None if it is unreadable/corrupt.

    None is the readable=False signal validate_batch's frame is built on -- an unreadable image
    is a normal finding for the gate to report, not an exception to propagate.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # cheap corruption check; invalidates `img` for further reads
        with Image.open(path) as img:  # re-open: verify() leaves the handle unusable
            return image_metadata(img)
    except UNREADABLE_EXCEPTIONS:
        return None
