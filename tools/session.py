"""Drive a session from the command line, the way the demo goes.

  tools/session.py new <name> <song.wav> <stems_dir>
  tools/session.py invite <name> "Invite a bassist"
  tools/session.py say <name> <musician> "too busy, I don't believe it yet"
  tools/session.py keep|back <name> <musician>
  tools/session.py mix <name> out.mp3
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from producer.session import Session  # noqa: E402

cmd, name, *rest = sys.argv[1:]
if cmd == "new":
    s = Session.create(name, Path(rest[0]), Path(rest[1]))
    print(json.dumps(s.state["song_map"], indent=1)[:1500])
    sys.exit()
s = Session(name)
if cmd == "invite":
    mid = s.invite(rest[0])
    m = s.state["musicians"][mid]
    t = m["takes"][-1]
    print(mid, "|", m["character"]["one_line"], "\nPLAN:", m["character"]["plan"], "\nSAYS:", t["to_producer"],
          "\nENGINEER ROUNDS:", len(t["engineer"]), [[v.get("pass") for v in r] for r in t["engineer"]], "\n", t["audio"])
elif cmd == "say":
    instr = s.direct(rest[0], rest[1])
    t = s.state["musicians"][rest[0]]["takes"][-1]
    print("HEARD AS:", instr["heard_as"], "\nINSTRUCTION:", instr["instruction"], "\nREPLY:", instr["reply_to_producer"],
          "\nSAYS:", t["to_producer"], "\nENGINEER:", [[v.get("pass") for v in r] for r in t["engineer"]], "\n", t["audio"])
elif cmd == "keep":
    s.keep(rest[0]); print("kept", rest[0])
elif cmd == "back":
    s.send_back(rest[0]); print("sent back", rest[0])
elif cmd == "mix":
    print(s.final_mix(Path(rest[0])))
