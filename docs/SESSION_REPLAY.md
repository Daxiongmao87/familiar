# Headless session replay

Run the deterministic replay regressions with:

```bash
.venv/bin/python -m pytest tests/regression/test_session_identity_replay.py -q
```

`tests/session_replay.py` provides `SessionReplay`, an `AudioSource` with
independent timestamped Discord events and transcript completion events.
It accepts an injected clock and async sleep function. The deterministic
tests advance a virtual clock instead of sleeping or connecting to Discord.

The identity regression registers the actual `VoicePresence.start` callback
on a fake Discord client. It delivers speaking events through that callback,
then feeds finals into `SessionEngine._on_stream_final`. OOC mode isolates
transcription and attribution from inference and card generation. Expected
identities are asserted separately from the event stream.

Covered scenarios include a missing callback member resolved through the
voice registry, a reassigned SSRC, overlapping speaking windows, a delayed
transcript completion, and an unresolved stream. The overlap assertion pins
the current dominant-speaker behavior; it does not claim word-level diarization.

## Real STT integration

Construct `SessionReplay` with a mono PCM16 16 kHz `wav_path`, Discord events,
and the default real monotonic clock/sleep. Run its producer concurrently
with `engine.consume_source(replay)`. This feeds the actual streaming adapter
and server without a desktop audio device. Use OOC mode when measuring only
transcription and attribution. Audio mode rejects injected transcript finals,
so expected text cannot accidentally become actual STT output.

The WAV is currently loaded into a chunk timeline: use short clips, not a
whole episode. The fixture audio and independent reference annotations must
share the same recording start, including leading silence. Final-event
fixtures specify `start`, `end`, `text`, and completion time `at`; speaking
events specify their own independent timing and callback payloads.

Use `word_error_rate(reference, actual)` for normalized word edit distance;
compare published transcript user IDs against independently annotated speaker
windows. Segment alignment and real-STT scoring still need an annotated audio
fixture. No real Critical Role audio accuracy result has been established by
the deterministic tests.

The existing `cr2e2_3h14m29s_crownsguard.json` is a transcript replay fixture,
not reviewed audio-aligned speaking-window ground truth. Do not derive both
mock speaking events and expected identities from its labels and report that
as independent validation. Audio assets remain external/gitignored.

## Offline real-audio diagnostic

`tools/check_recorded_identity.py WAV MODEL ANNOTATION [DECODED_JSON]`
decodes recorded speech with a local faster-whisper model on CPU, then sends
its timestamped segments through the production attribution callback with
mock Discord speaking events. The annotation is a JSON list of `start`,
`end`, `name`, and `text` objects. Optional decoded JSON contains recognizer
segments (`start`, `end`, `text`) captured separately; cached mode does not
require faster-whisper installed in the application environment.

This tests real recognizer output plus attribution, not the streaming STT
transport or production latency. Coarse transcript timestamps can produce
wrong speaker assignments near turn boundaries. Review boundaries against
audio before interpreting those assignments as algorithm failures or using
them as accuracy scores. WER is relative to the supplied reference only;
edited transcripts and omitted crosstalk affect that measurement. Keep audio,
annotations, decoded output, and diagnostic results out of source control.
