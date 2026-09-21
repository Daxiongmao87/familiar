"""Run actual recorded speech through Whisper and Familiar attribution.

CPU integration diagnostic, not a streaming-latency benchmark. Speaker
windows come from a separately supplied JSON annotation, never the recognizer.
Usage: python tools/check_recorded_identity.py WAV MODEL ANNOTATION [DECODED_JSON]
Annotation: [{"start": seconds, "end": seconds, "name": str, "text": str}].
DECODED_JSON may contain previously captured recognizer output; it must not
contain the reference transcription. This diagnostic prints observations,
not an independent speaker-accuracy verdict. Annotation timing must be
reviewed against audio before interpreting attribution mismatches.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmd.config import load_config_dict
from dmd.pipeline import SessionEngine
from dmd.speaking_tracker import SpeakingTracker
from dmd.voice_presence import VoicePresence
from tests.session_replay import ReplayEvent, SessionReplay, word_error_rate


async def main():
    """Decode real audio, replay independent speaker events, print evidence."""
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "Usage: check_recorded_identity.py WAV MODEL ANNOTATION [DECODED_JSON]"
        )
    wav, model_path, annotation = sys.argv[1:4]
    refs = json.loads(Path(annotation).read_text())
    if len(sys.argv) > 4:
        decoded = json.loads(Path(sys.argv[4]).read_text())
    else:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=8)
        segments, _ = model.transcribe(wav, language="en", word_timestamps=True)
        decoded = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    print(json.dumps({"decoded": decoded}), flush=True)
    names = list(dict.fromkeys(r["name"] for r in refs))
    members = {100000000000000000 + i: SimpleNamespace(
        id=100000000000000000 + i, display_name=name) for i, name in enumerate(names)}
    ids = {member.display_name: uid for uid, member in members.items()}
    mapping = {1502 + i: uid for i, uid in enumerate(members)}
    guild = SimpleNamespace(voice_client=SimpleNamespace(_ssrc_to_id=mapping),
                            get_member=members.get)
    now = [100.0]

    class Client:
        """Library boundary; production bot callback remains unmodified."""
        def __init__(self, **kwargs):
            pass
        def event(self, fn):
            setattr(self, fn.__name__, fn)
            return fn
        def get_guild(self, uid):
            return guild
        async def start(self, token):
            pass

    tracker = SpeakingTracker()
    bot = VoicePresence("fixture", 1, next(iter(members)), tracker)
    with patch.dict(sys.modules, discord=SimpleNamespace(
        Client=Client, Intents=SimpleNamespace(none=lambda: SimpleNamespace()))):
        await bot.start()
    cfg = load_config_dict({"models": {"synthesis": {
        "base_url": "http://unused", "model_id": "unused"}, "stt": {}}})
    published = []
    engine = SessionEngine(cfg, None, None, [], None, None, published.append,
                           speaking_tracker=tracker)
    engine.set_ooc(True)
    events = []
    for ref in refs:
        ssrc = next(k for k, v in mapping.items() if v == ids[ref["name"]])
        for at, state in [(ref["start"], 1), (ref["end"], 0)]:
            events.append(ReplayEvent(at, "discord", dict(ssrc=ssrc, state=state)))
    for segment in decoded:
        events.append(ReplayEvent(segment["end"] + .1, "final", segment))

    async def discord_sink(event):
        await bot._client.on_member_speaking_state_update(
            None, event["ssrc"], event["state"])

    async def sleep(seconds):
        now[0] += seconds

    try:
        with patch("dmd.speaking_tracker.time", SimpleNamespace(monotonic=lambda: now[0])):
            replay = SessionReplay(events, discord_sink, engine._on_stream_final,
                                   clock=lambda: now[0], sleep=sleep)
            await replay.run()
        transcripts = [p for p in published if p["type"] == "transcript"]
        print(json.dumps({"mode": "offline_decode_mock_discord",
                          "cached_decode": len(sys.argv) > 4,
                          "speaker_accuracy_verified": False,
                          "transcripts": transcripts,
                          "word_error_rate": word_error_rate(
                              " ".join(r["text"] for r in refs),
                              " ".join(s["text"] for s in decoded)),
                          "stream_ids_as_people": any(
                              p["user_id"] in {str(k) for k in mapping} for p in transcripts)
                          }, indent=2), flush=True)
    finally:
        await engine.aclose()
        await bot._task


if __name__ == "__main__":
    asyncio.run(main())
