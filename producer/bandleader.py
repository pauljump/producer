"""The bandleader as an agent: Muse Wonder runs the session through its own tools.

The producer talks to the band as a whole ("give me a bass and drums, keep the verses sparse,
let the la-la part lift"). Wonder decides who to hire, what to tell each player, and in what
order, and delegates each take to that musician's own Wonder sub-agent. Independent takes run
in parallel (Wonder's parallel tool calls). Each take still goes through the engineer before the
producer hears it. The producer's taste is final: Wonder reports back and waits.
"""
import json
from concurrent.futures import ThreadPoolExecutor

from . import band, models

SYSTEM = """You are the bandleader on a recording session. The producer knows nothing about music,
only what they like, and is decisive about it. Your job is to turn what they say into the right
players playing the right parts, then hand the takes back for their verdict.

You have tools. Hire players the song needs and the producer asked for. Give each player a
specific musical instruction (density, register, rhythm, dynamics, where in the song), in the
spirit of what the producer said; read taste generously ("I don't believe it" usually means
generic or effortful). Players you brief in the same turn play at the same time. When the takes
are back, call report_to_producer once with a short, plain reply (no theory). Never pretend a
take is kept: only the producer keeps or sends back.

SONG MAP: {song_map}"""

TOOLS = [
    {"type": "function", "function": {"name": "band_status",
     "description": "Who is on the session: the original performers, hired players, their last take and whether the producer kept it.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "hire",
     "description": "Hire a session player. They are cast with a character, look up how their influences play, then wait for a brief.",
     "parameters": {"type": "object", "properties": {"who": {"type": "string", "description": "e.g. 'a bassist who stays out of the singer's way'"}},
                    "required": ["who"]}}},
    {"type": "function", "function": {"name": "brief",
     "description": "Send one player into the booth with an instruction. They listen, write their part, check it, and the engineer checks it before the producer hears it. Returns what they played.",
     "parameters": {"type": "object", "properties": {
         "musician_id": {"type": "string"}, "instruction": {"type": "string"},
         "where": {"type": "string"}, "keep": {"type": "string", "description": "what must not change from their last take"}},
         "required": ["musician_id", "instruction"]}}},
    {"type": "function", "function": {"name": "report_to_producer",
     "description": "End your turn with a short plain reply to the producer.",
     "parameters": {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}}},
]


def _status(session) -> dict:
    st = session.state
    return {"originals": [{k: o.get(k) for k in ("name", "role", "what_they_played")} for o in st.get("originals", [])],
            "players": [{"id": mid, "name": m["character"]["name"], "instrument": m["character"]["instrument"],
                         "kept": m.get("kept") is not None,
                         "last_take": (m["takes"][-1].get("to_producer") if m["takes"] else None),
                         "producer_sent_back_last": bool(m["takes"] and m["takes"][-1].get("sent_back"))}
                        for mid, m in st["musicians"].items()]}


def ask_band(session, words: str, max_turns: int = 8) -> dict:
    """One exchange: the producer speaks, the bandleader works the band, and reports back."""
    msgs = [{"role": "system", "content": SYSTEM.format(song_map=json.dumps(session.state["song_map"]))},
            {"role": "user", "content": f'The producer says: "{words}"'}]
    actions = []
    session.log("producer to band", words=words)
    for _ in range(max_turns):
        resp = models.chat("meta", models.WONDER, msgs, purpose="bandleader agent", estimate_usd=0.3,
                           max_tokens=16000, tools=TOOLS)
        m = resp.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            msgs.append({"role": "user", "content": "Use your tools; finish with report_to_producer."})
            continue
        calls = [(c, json.loads(c.function.arguments or "{}")) for c in m.tool_calls]
        for c, a in calls:
            if c.function.name == "report_to_producer":
                session.log("bandleader reply", reply=a.get("reply"), actions=actions)
                session.save()
                return {"reply": a.get("reply", ""), "actions": actions}

        def run(item):
            c, a = item
            name = c.function.name
            if name == "band_status":
                return _status(session)
            if name == "hire":
                mid = session.hire(a["who"])
                actions.append({"hire": mid})
                ch = session.state["musicians"][mid]["character"]
                return {"musician_id": mid, "name": ch["name"], "instrument": ch["instrument"], "plan": ch.get("plan"),
                        "looked_up": (ch.get("research") or {}).get("text", "")[:800]}
            if name == "brief":
                mid = a["musician_id"]
                if mid not in session.state["musicians"]:
                    return {"error": f"no player {mid}; call band_status"}
                direction = f'{a["instruction"]} Where: {a.get("where", "the whole song")}. Keep: {a.get("keep", "")}.'
                m_ = session.state["musicians"][mid]
                previous = m_["takes"][-1] if m_["takes"] else None
                session._play(mid, direction=direction if previous else None, previous=previous, said=words,
                              translation={"instruction": a["instruction"], "where": a.get("where"),
                                           "reply_to_producer": None}, plan_note=None if previous else a["instruction"])
                t = m_["takes"][-1]
                actions.append({"brief": mid, "instruction": a["instruction"]})
                return {"played": t.get("to_producer"), "engineer_rounds": len(t.get("engineer", [])),
                        "engineer_passed": all(v.get("pass") is not False for v in (t["engineer"][-1] if t.get("engineer") else []))}
            return {"error": "unknown tool"}

        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(run, calls))
        for (c, _), r in zip(calls, results):
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(r, default=str)})
    session.save()
    return {"reply": "The band is still working on it.", "actions": actions}
