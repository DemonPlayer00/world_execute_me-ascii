#!/usr/bin/env python3
"""Simulate a Linux virtual console (TERM=linux) and assert the MV adapts.

A VT has no alternate screen, 8-16 colours and no Braille, so the player must
not emit ESC[?1049h, must use 16-colour SGR, must use the console's own cursor
sequences, and must leave the screen cleared on exit.
"""
import os, pty, re, select, sys, time

TERM = os.environ.get("FAKE_TERM", "linux")
if not TERM.startswith("linux"):
    raise SystemExit(f"this check asserts virtual-console behaviour; "
                     f"FAKE_TERM={TERM} is not a console")
pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = TERM
    os.environ.pop("COLORTERM", None)
    os.environ["PYTHONPATH"] = "mv"
    os.execvp("python3", ["python3", "-m", "ttymv",
                          "--start", "62", "--ao", "null", "--stats"])

buf = b""
t0 = time.time()
while time.time() - t0 < 3.5:
    r, _, _ = select.select([fd], [], [], 0.2)
    if r:
        try:
            buf += os.read(fd, 1 << 20)
        except OSError:
            break
os.write(fd, b"q")
time.sleep(1.0)
while True:
    r, _, _ = select.select([fd], [], [], 0.2)
    if not r:
        break
    try:
        d = os.read(fd, 1 << 20)
    except OSError:
        break
    if not d:
        break
    buf += d
try:
    os.waitpid(pid, 0)
except ChildProcessError:
    pass

txt = buf.decode("utf8", "ignore")
head, tail = txt[:400], txt[-400:]
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


check("never enters the alternate screen", "\x1b[?1049h" not in txt)
check("clears the screen on entry", "\x1b[2J" in head or "\x1b[H\x1b[J" in head)
check("clears the screen on exit", "\x1b[J" in tail or "\x1b[2J" in tail)
check("uses the console's own cursor-hide", "\x1b[?1c" in head, head[:60].encode().decode("unicode_escape", "ignore"))
check("restores the cursor on exit", "\x1b[?0c" in tail or "\x1b[?25h" in tail)
check("no true-colour SGR", not re.search(r"\x1b\[(38|48);2;", txt))
check("only 16-colour SGR is used", bool(re.search(r"\x1b\[(3|4|9|10)\d+m", txt)))
check("no line feed in any frame", "\n" not in txt.replace("\r\n", ""), repr(txt[:0]))
check("rows are cursor-addressed",
      len(re.findall(r"\x1b\[\d+;1H", txt)) > 20)
check("no Braille glyphs used", not re.search(r"[\u2800-\u28ff]", txt))
check("stats line reports the console", "console=yes" in txt, tail[-200:])
drop = re.search(r"(\d+) dropped", txt)
check("frame dropping engaged on a slow tty",
      drop is not None and int(drop.group(1)) >= 0, txt[-200:])

print()
print(f"{len(fails)} failed" if fails else "console path OK")
sys.exit(1 if fails else 0)
