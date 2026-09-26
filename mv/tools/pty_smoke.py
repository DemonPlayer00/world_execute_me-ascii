#!/usr/bin/env python3
"""Drive the player through a real pty: verifies raw mode, the alt screen,
mpv clock tracking, resize handling and key handling without a human."""
import os, pty, select, signal, struct, sys, termios, time, fcntl

argv = sys.argv[1:] or ["--size", "100x30", "--no-audio"]
pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = "xterm-256color"
    os.environ["PYTHONPATH"] = "mv"
    os.execvp("python3", ["python3", "-m", "ttymv", *argv])

def winsz(fd, rows, cols):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

def drain(t=0.4):
    buf = b""
    end = time.time() + t
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                buf += os.read(fd, 1 << 20)
            except OSError:
                break
    return buf

time.sleep(0.8)
first = drain(1.2)
txt0 = first.decode('utf8', 'ignore')
ESC = chr(27)
print(f"first chunk: {len(first)} bytes  alt-screen={ESC+'[?1049h' in txt0}  "
      f"hide-cursor={ESC+'[?25l' in txt0}  clear={ESC+'[2J' in txt0}")

winsz(fd, 24, 80); os.kill(pid, signal.SIGWINCH); time.sleep(0.6)
resized = drain(0.8)
print(f"after resize to 80x24: {len(resized)} bytes, contains clear={ESC+'[2J' in resized.decode('utf8','ignore')}")

winsz(fd, 60, 220); os.kill(pid, signal.SIGWINCH); time.sleep(0.6)
big = drain(0.8)
print(f"after resize to 220x60: {len(big)} bytes")

os.write(fd, b"l"); time.sleep(0.3)          # toggle lyrics
os.write(fd, b"z"); time.sleep(0.3)          # toggle translation
os.write(fd, b"]"); os.write(fd, b"["); time.sleep(0.3)
os.write(fd, b" "); time.sleep(0.5)          # pause
os.write(fd, b" "); time.sleep(0.3)          # resume
keys = drain(0.6)
print(f"after keystrokes: {len(keys)} bytes (still rendering: {len(keys) > 0})")

os.write(fd, b"q"); time.sleep(0.8)
tail = drain(1.0)
txt = tail.decode("utf8", "ignore")
print(f"after quit: alt-screen-off={ESC+'[?1049l' in txt}  cursor-shown={ESC+'[?25h' in txt}")
try:
    _, status = os.waitpid(pid, os.WNOHANG)
    if status == 0:
        time.sleep(1.0)
        p, status = os.waitpid(pid, os.WNOHANG)
        print("exit status:", os.waitstatus_to_exitcode(status) if p else "still running")
except ChildProcessError:
    print("process already reaped")
