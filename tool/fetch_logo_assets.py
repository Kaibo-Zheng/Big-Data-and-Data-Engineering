"""Fetch official-site logo assets used by the report figures.

The script keeps the final plotting pipeline offline-friendly: it downloads the
selected homepage/brand-page assets once, normalizes them to small PNG files,
and records where each asset came from.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageFile


ROOT = Path(__file__).resolve().parents[1]
LOGO_DIR = ROOT / "Illustration" / "logos"
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class LogoAsset:
    family: str
    homepage: str
    source: str
    filename: str
    note: str


ASSETS = [
    LogoAsset(
        "OpenAI",
        "https://openai.com/brand/",
        "https://openai.com/apple-icon.png?apple-icon.02be~wu.bus9e.png",
        "openai.png",
        "OpenAI brand page apple-touch icon.",
    ),
    LogoAsset(
        "Alibaba",
        "https://www.alibabacloud.com/",
        "https://img.alicdn.com/tfs/TB1ugg7M9zqK1RjSZPxXXc4tVXa-32-32.png_.webp",
        "alibaba.png",
        "Alibaba Cloud homepage shortcut icon.",
    ),
    LogoAsset(
        "DeepSeek",
        "https://www.deepseek.com/en/",
        "https://cdn.deepseek.com/logo.png?x-image-process=image%2Fresize%2Cw_1920",
        "deepseek.png",
        "DeepSeek homepage logo image.",
    ),
    LogoAsset(
        "ByteDance",
        "https://www.bytedance.com/",
        "data:image/vnd.microsoft.icon;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAMAAABOo35HAAACFlBMVEUAAACp3f8yW7U8jf8zW7U/3vJA5P8AydMzW7VPlP955908jP8AydM8jf8yWrV5590yWrV45908jf8yW7UyW7UyW7V55909jf88jf8yW7UzW7V56N49jf80XLY+j/80XbY/lf8zZr5Ilf9GXbkA2+RCY70zZsx459x5590AydI8jP8AydN4590AydM8jf8AydM8jf88jf94590AyNM8jf8AydN5590AydM8jf8AyNIAytR5594zW7V56N09jf955948jf8AydMAytN65948jf956N56594AytMAy9Q9jv976d8AytYAy9Z66N566uE/jv8AztUAy9cAytV56OSD798A0NYAydM8jP95590yW7R45twAydN45909jP8yWrU8jf945t0AydIyWrV5590AydIyW7UzWrU8jf8AydIzW7UzWrV459145908jf8yW7UAydM8jP8AydM8jf9459155909jf8zW7UAydQAytMAydN56N0AydMzXLYAydM0W7UAydQ8jv8+jv8Ay9QzW7Y9jf965t176N80W7YAydN56N156d4yW7U0XLc+jv8+kP8Ay9VAj/81XbgAzNQ1XbY5XbZAkf9+7eQ7Yrp96+FDlv+F6emI/+4AydN55t15594AydM+jf8+jf8+jv8AydQ1W7QzW7UAydR75942XbUyWrR86OCA8uYAztqA8eN45twyWrQAyNI8jP8bed+KAAAArnRSTlMAAfb9+wUD8HMJ/fr59+jb1dTLt6eamY+Jg3xXTkhHOhIRDQsJBwX59vbr4+Dc1tXRxcC7ubSqoJuak5CKiIN+enlyb21lXVFEQj09NzAnJiQdGBUPDvzx8O/s6Obl397OzsvIyMXEwcG/vbiysbGuqqeknZSTkI2Gf3ZraGViX15YWFVTUk9OS0pEQUA8MzAtLSomHx4bGhkXCwfApKCAdGdjXlxaWTY0MyEUFBKNTPVmAAAFHElEQVR42uzX105UYRSG4YUyiCgqKvbeK3ZFsaCCXUHE3kvU2FvU2EussfcSY2KiiSUDc4fuTcl/xr8/ZuZk7/e9g/UcfcuIiIiIiIiIiIiIiMhf7wFDPxlFKx1UdunBX6NoWGEHqp40GnmwXCWHlzz9YuTBck2cvWzdVyMPlmtSAJYy8mE5sJPXNhYYebBcO06tBMyP5do1v3aTkcPytedb7R8jh+Vr2sVVHyzxpaMXfEZbLNGltcqqHjba8LGWzNJyJdZUeHTws/GWvNJ61hTW5cSQkRMsWXUWqxWsX/HvJC2xTmK5dp4pfp0UsCywXL3633hrCShrLAd2653FvBxgufaV//xoMS5HWK4+5fdHW0zLJZYDu/wolmC5xXIVHlocv6GfcyxX/IZ+HrHiN/TziuWGvsWiLLCULBaBBVZLYAmBJQSWEFhCYAmBJQSWEFhCYAmBJQSWEFhCYAmBJQSWEFhCYAmBJQSWEFhCYAmBJQSWEFhCYAmBJQSWEFhCYAmBJQSWEFhCYAmBJQSWEFhCYOW1erCitfn2gqnNYPlruLdwenMQWJ7+rV40I3QCy9PWx1cOhkZgeRq35uqRbYEPWD6o9d+P9wxxwOq41IblcyaHMGB1XNHLFfO6hShgeXpz/WwLFFie6m+e3x1igBVlmgeB5anhV+s0B8tX2zQHK0rB8WCBZQYWWGCBBRZYYIEFFlhggQUWWGCBBRZYYIGVZKyKgcNGmYEVCSuTyXTtO2jEGLCiYYX1mLV0bXewfFiu0mPVz1NgebEc2NzqF0Vg+bBc20/XvCoAy4flmnKupg4sL5Zr74UfdWB5sVz7K++8B8theauovPsZLIflbWYw9MGKiNU+9MGKlBv6YEWvFCwhsMACCyywwAILLLDAAgsssMACCyywwAILLLDAAgsssMACCyywwAILLLDAAgsssMACCyywwAILLLDAAgsssMACCyywwAILLLDAAgsssMD6325d7cQVhWEY/ocWn9JSoO5GhTqlSo2Wurt72qbu3lRwgruEQCCBE+6RHQ7YR7DWP8OeWSHvewnPwZcPLLDAAgsssMACCyywwAILLLDAAgsssMACCyywwAILLLCmGtbhHclg+Vimkk4f2r4OLLEu8dS3Z2Gw7Ft1/OvjMFj2Xan48uAqWAqwY3s2g6Xo4r/dG8FSdKF/5wawFJ3/9e4WWIoaf7xJBsu+UJ139MHyszr6YKmOfncYLM3RB0sRWGDZBBZYYIEFFlhggQUWWGBFgpWQBZYd1PMFS9cIWMamP5lfvVq8wDJAPdy3eKV4gWVo1ufy5eIFlhigcv6PQoFlwLqbM9AmXmAZsDKyfzeLX6BY6U/FJjexMrJ/NolfoFjTZs9bkipWuYd143VKQ0iiyh5qy9xFK8Q6t7ASslKGfKiAsTZ9KlkmqtzB8q95DLDu7frbIXEvIiz/mscC6877Py3iRJFg+dc8cKz1b/vOiTPprdJk0hsH6tWRenEqpVTmx9JhkeCxruX31obEtRRQM+csvCTq9Fgz8g/WJImLWULl9RS3ijo9VvrW/SdSxdUsoG4XHD0rgRb3az45WDcLitRQ+uJ+zaPHuv6y6ExsRjbu1zw6rLUvvg+6ObKqgsfK3XbgZKKQESv3UWFVl5ARK+1+YWWnkBkrc2/ZZaGJG7vmZC7vQ3G7EBERERERERERERER0dRtBK5Q857p1uutAAAAAElFTkSuQmCC",
        "bytedance.png",
        "ByteDance homepage shortcut icon data URI.",
    ),
    LogoAsset(
        "Moonshot",
        "https://www.moonshot.ai/",
        "https://statics.moonshot.cn/moonshot-ai/favicon.ico",
        "moonshot.png",
        "Moonshot AI homepage favicon.",
    ),
    LogoAsset(
        "Google",
        "https://www.google.com/",
        "https://www.google.com/images/branding/googlelogo/1x/googlelogo_white_background_color_272x92dp.png",
        "google.png",
        "Google homepage branding image.",
    ),
    LogoAsset(
        "Anthropic",
        "https://www.anthropic.com/",
        "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/681d52619fec35886a7f1a70_favicon.png",
        "anthropic.png",
        "Anthropic homepage favicon.",
    ),
    LogoAsset(
        "xAI",
        "https://x.ai/",
        "https://x.ai/favicon.ico?favicon.075p690g1s4oh.ico?dpl=4a7fa70dc851903bbdfa247a9c773e66d1959076",
        "xai.png",
        "xAI homepage favicon.",
    ),
]


def fetch_bytes(source: str) -> bytes:
    if source.startswith("data:"):
        _, payload = source.split(",", 1)
        return base64.b64decode(payload)
    response = requests.get(
        source,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (CCKS2026 report logo fetcher)"},
    )
    response.raise_for_status()
    return response.content


def trim_near_white(im: Image.Image, threshold: int = 248) -> Image.Image:
    rgba = im.convert("RGBA")
    pix = rgba.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = pix[x, y]
            if a > 8 and min(r, g, b) < threshold:
                xs.append(x)
                ys.append(y)
    if not xs:
        return rgba
    pad = 3
    box = (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(rgba.width, max(xs) + 1 + pad),
        min(rgba.height, max(ys) + 1 + pad),
    )
    return rgba.crop(box)


def normalize_logo(raw: bytes, output: Path, canvas_size: int = 256) -> None:
    im = Image.open(io.BytesIO(raw))
    if getattr(im, "is_animated", False):
        im.seek(0)
    im = trim_near_white(im)
    max_side = int(canvas_size * 0.76)
    scale = min(max_side / im.width, max_side / im.height)
    new_size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
    im = im.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    x = (canvas_size - im.width) // 2
    y = (canvas_size - im.height) // 2
    canvas.alpha_composite(im.convert("RGBA"), (x, y))
    canvas.save(output)


def main() -> None:
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Logo Asset Sources",
        "",
        "Assets are downloaded from official homepages or brand pages for report visualization.",
        "They are used only as small identification marks in dataset-analysis figures.",
        "",
        "| Family | Homepage | Asset source | Local file | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for asset in ASSETS:
        out = LOGO_DIR / asset.filename
        raw = fetch_bytes(asset.source)
        normalize_logo(raw, out)
        source_display = asset.source
        if source_display.startswith("data:"):
            source_display = "embedded data URI from homepage shortcut icon"
        lines.append(
            f"| {asset.family} | {asset.homepage} | {source_display} | `{out.relative_to(ROOT)}` | {asset.note} |"
        )
        print(f"wrote {out.relative_to(ROOT)}")
    (LOGO_DIR / "logo_sources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {(LOGO_DIR / 'logo_sources.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
