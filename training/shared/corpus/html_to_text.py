"""
html_to_text.py
Converts 'Selected Poems - Guillaume Apollinaire & A. S. Kline.html'
to a clean plain-text file suitable for language model training.

What it keeps:
  - Poem titles (h2 / h3 headings that are poem/section names)
  - Verse lines
  - Blank lines between stanzas (from &nbsp; paragraphs)

What it strips:
  - Image captions / attribution lines (p.small-font)
  - The "Notes to the Bestiary" section and everything after it
  - The "Index of First Lines" ul list
  - Source-collection subtitles like "(Alcools: Crépuscule)"
  - Diamond / glyph separator lines
"""

import re
from html.parser import HTMLParser
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────────────

INPUT_FILE  = Path(__file__).parent / "Selected Poems - Guillaume Apollinaire & A. S. Kline.html"
OUTPUT_FILE = Path(__file__).parent / "apollinaire_poems.txt"

# Substring(s) that appear in headings that signal end of poem content
STOP_HEADING_KEYWORDS = ["notes to the bestiary", "index of first lines"]

# Source-collection lines like "(Alcools: Crépuscule)"
COLLECTION_RE = re.compile(r"^\(.*\)\s*$")

# Diamond / special-glyph dividers  ◇◇◇◇
GLYPH_RE = re.compile(r"^[\u25c7\u25c6\u2666\u2b26\u25ca\s\u00a0]+$")

# ── HTML parser ───────────────────────────────────────────────────────────────

class PoemParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []

        # block-level state  (h2 / h3 / p)
        self._block_tag   = None   # current open block tag
        self._block_class = ""     # class of the current block tag
        self._block_text  = []     # accumulated text for the current block

        self._skip      = False    # True once we hit a stop-heading
        self._in_list   = False    # inside the Index <ul>

    # ── tag open ──────────────────────────────────────────────────────────────
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag in ("h2", "h3", "p"):
            # start of a new block — capture its class
            self._block_tag   = tag
            self._block_class = attrs_dict.get("class", "")
            self._block_text  = []

        elif tag == "ul":
            self._in_list = True

        elif tag == "hr":
            if not self._skip:
                self.lines.append("")

        # inner inline tags (em, a, strong, br…): don't touch block state

    # ── tag close ─────────────────────────────────────────────────────────────
    def handle_endtag(self, tag):
        if tag == "ul":
            self._in_list = False
            return

        if self._in_list:
            return

        if tag not in ("h2", "h3", "p"):
            return   # ignore closing inline tags

        if tag != self._block_tag:
            # mis-matched close (shouldn't happen in valid HTML, but be safe)
            return

        text = "".join(self._block_text).strip()
        cls  = self._block_class

        # reset block state
        self._block_tag   = None
        self._block_class = ""
        self._block_text  = []

        if not text:
            return

        # ── stop condition ───────────────────────────────────────────────────
        if tag in ("h2", "h3"):
            if any(kw in text.lower() for kw in STOP_HEADING_KEYWORDS):
                self._skip = True
                return

        if self._skip:
            return

        # ── headings ─────────────────────────────────────────────────────────
        if tag in ("h2", "h3"):
            self.lines.append("")
            self.lines.append(text)
            self.lines.append("")
            return

        # ── paragraphs ───────────────────────────────────────────────────────

        # caption / attribution: class contains "small-font"
        if "small-font" in cls:
            return

        # blank-line stanza separator  (&nbsp; → '\xa0')
        cleaned = text.replace("\xa0", "").strip()
        if not cleaned:
            self.lines.append("")
            return

        # source-collection subtitle: "(Alcools: Signe)"
        if COLLECTION_RE.match(text):
            return

        # glyph-only divider lines
        if GLYPH_RE.match(cleaned):
            self.lines.append("")
            return

        # verse / prose line
        self.lines.append(text)

    # ── raw character data ────────────────────────────────────────────────────
    def handle_data(self, data):
        # accumulate into current open block if one exists
        if self._block_tag is not None:
            self._block_text.append(data)


# ── post-processing ───────────────────────────────────────────────────────────

def collapse_blank_lines(lines: list[str]) -> list[str]:
    """Collapse 3+ consecutive blank lines into at most 2."""
    result    = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)
    while result and result[0]  == "": result.pop(0)
    while result and result[-1] == "": result.pop()
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    html = INPUT_FILE.read_text(encoding="utf-8")

    parser = PoemParser()
    parser.feed(html)

    lines = collapse_blank_lines(parser.lines)
    output = "\n".join(lines) + "\n"
    OUTPUT_FILE.write_text(output, encoding="utf-8")

    print(f"Done. {len(lines)} lines written to {OUTPUT_FILE.name}\n")
    print("=== FIRST 40 LINES ===")
    print("\n".join(lines[:40]))
    print("\n=== LAST 20 LINES ===")
    print("\n".join(lines[-20:]))


if __name__ == "__main__":
    main()
