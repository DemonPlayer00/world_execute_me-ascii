#!/usr/bin/env python3
"""The MV must end by itself when the music ends -- no keypress involved."""
import os, pty, select, sys, time

pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = "xterm-256color"
    os.environ["PYTHONPATH"] = "mv"
    os.execvp("python3", ["python3", "-m", "ttymv",
                          "--start", "205.5", "--ao", "null", "--size", "80x24"])

t0 = time.monotonic()
result = "DID NOT EXIT within 25s"
exited = False
while time.monotonic() - t0 < 25:
    select.select([fd], [], [], 0.25)
    try:
        os.read(fd, 1 << 20)
    except OSError:
        pass
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        result = (f"exited on its own after {time.monotonic() - t0:.1f}s  "
                  f"code={os.waitstatus_to_exitcode(status)}")
        exited = True
        break
if not exited:
    os.kill(pid, 9)
    os.waitpid(pid, 0)
print(result)
