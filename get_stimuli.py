"""Download a set of public-domain images for the gaze heatmap slideshow.

Classic pictures from eye-tracking research and art, plus two generated tasks.
Each painting is found through the Wikimedia Commons API, its licence is
checked (only public domain is accepted), and the source is written to
stimuli/CREDITS.md.

Usage:
  python get_stimuli.py
  python gaze_demo.py --heatmap --stimulus stimuli/ --seconds 12
"""

import json
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np


OUT = Path("stimuli")
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "GazeMap-stimuli/1.0 (student eye-tracking project; contact via GitHub fammad)"}
WIDTH = 1800

# (file name, Commons search, words the file title must contain, why it is interesting)
PAINTINGS = [
    ("01_repin_unexpected_visitors", "Repin They Did Not Expect Him", ["repin"],
     "The painting from Yarbus's 1967 study: gaze depends on the question you ask. "
     "Try it twice: 'how rich is the family?' then 'how long was he away?'"),
    ("02_mona_lisa", "Mona Lisa Leonardo da Vinci", ["mona", "lisa"],
     "Faces pull the eyes to the eyes and mouth."),
    ("03_girl_with_a_pearl_earring", "Girl with a Pearl Earring Vermeer", ["pearl"],
     "One face, strong gaze cue: does your heatmap follow her eyes or the earring?"),
    ("04_duck_rabbit", "Duck-Rabbit illusion", ["duck", "rabbit"],
     "Ambiguous figure: look at the left side for the duck, the right for the rabbit."),
    ("05_wife_and_mother_in_law", "My Wife and My Mother-in-Law Hill 1915", ["mother"],
     "Ambiguous figure: young woman or old woman depends on where you look."),
    ("06_netherlandish_proverbs", "Netherlandish Proverbs Bruegel", ["proverbs"],
     "A crowded scene with about 100 small stories: free search, many scattered fixations."),
    ("07_hunters_in_the_snow", "Hunters in the Snow Bruegel", ["hunters"],
     "Strong diagonal composition that leads the eye into the distance."),
    ("08_grande_jatte", "Sunday Afternoon Island of La Grande Jatte Seurat", ["jatte"],
     "Many people, calm layout: compare with the busy Bruegel."),
]


def api_search(query):
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
        "gsrnamespace": 6, "gsrlimit": 10, "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": WIDTH,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        pages = json.load(response).get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda page: page.get("index", 999))


def pick(pages, must_contain):
    for page in pages:
        title = page.get("title", "").lower()
        info = (page.get("imageinfo") or [{}])[0]
        licence = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
        if info.get("mime") not in ("image/jpeg", "image/png", "image/svg+xml"):
            continue  # SVGs are fine: the thumbnail URL is a rendered PNG
        if not all(word in title for word in must_contain):
            continue
        if "public domain" not in licence.lower() and not licence.lower().startswith("pd"):
            continue
        if any(word in title for word in ("detail", "crop", "fragment", "x-ray", "frame")):
            continue
        return page, info, licence
    return None


def download(url, path):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        data = response.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not an image")
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return image.shape


def make_reading_page(path):
    """A page of text: reading leaves left-to-right lines, an F-shape on skimming."""
    text = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having "
        "nothing to do: once or twice she had peeped into the book her sister was reading, but it had "
        "no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without "
        "pictures or conversations?' So she was considering in her own mind (as well as she could, for "
        "the hot day made her feel very sleepy and stupid), whether the pleasure of making a "
        "daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a "
        "White Rabbit with pink eyes ran close by her. There was nothing so very remarkable in that; "
        "nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! "
        "Oh dear! I shall be late!'"
    )
    page = np.full((1080, 1728, 3), 250, np.uint8)
    cv2.putText(page, "Down the Rabbit-Hole  (Lewis Carroll, 1865)", (110, 130),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (40, 40, 40), 2)
    for k, line in enumerate(textwrap.wrap(text, 72)):
        cv2.putText(page, line, (110, 230 + 62 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2)
    cv2.imwrite(str(path), page)


def make_odd_one_out(path, seed=7):
    """Visual search: one O hidden among many C's. The heatmap shows the search path."""
    rng = np.random.default_rng(seed)
    page = np.full((1080, 1728, 3), 245, np.uint8)
    rows, cols = 9, 15
    target = (rng.integers(rows), rng.integers(cols))
    for r in range(rows):
        for c in range(cols):
            x, y = 110 + c * 105, 120 + r * 108
            angle = int(rng.integers(0, 4)) * 90
            if (r, c) == target:
                cv2.circle(page, (x, y), 26, (40, 40, 40), 6)
            else:
                cv2.ellipse(page, (x, y), (26, 26), angle, 40, 320, (40, 40, 40), 6)
    cv2.putText(page, "Find the closed circle", (110, 1055), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 90, 90), 2)
    cv2.imwrite(str(path), page)


def main():
    OUT.mkdir(exist_ok=True)
    credits = ["# Stimulus credits", "",
               "Paintings: public domain, via Wikimedia Commons. Generated pages: made by get_stimuli.py.", ""]

    for name, query, must, why in PAINTINGS:
        path = OUT / f"{name}.jpg"
        try:
            found = pick(api_search(query), must)
            if found is None:
                print(f"skipped {name}: no public-domain match found")
                continue
            page, info, licence = found
            shape = download(info.get("thumburl") or info["url"], path)
            source = info.get("descriptionurl", "")
            print(f"saved {path}  ({shape[1]}x{shape[0]}, {licence})")
            credits.append(f"- **{path.name}**: {page['title']}. {licence}. Source: {source}  \n  {why}")
        except Exception as error:  # network or API problem: keep going with the rest
            print(f"skipped {name}: {error}")

    make_reading_page(OUT / "09_reading_alice.png")
    make_odd_one_out(OUT / "10_find_the_circle.png")
    credits.append("- **09_reading_alice.png**: text from Alice's Adventures in Wonderland (public domain). "
                   "Reading should give left-to-right line patterns.")
    credits.append("- **10_find_the_circle.png**: visual search task. The heatmap shows where you searched "
                   "before finding the target.")
    print("saved stimuli/09_reading_alice.png, stimuli/10_find_the_circle.png")

    (OUT / "CREDITS.md").write_text("\n".join(credits) + "\n")
    print("wrote stimuli/CREDITS.md")


if __name__ == "__main__":
    main()
