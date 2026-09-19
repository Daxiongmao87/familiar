"""Test the vendored server protocol without loading GPU models."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def server_class():
    """Load the socket processor independently of model startup."""
    path = Path(__file__).resolve().parents[2] / "services/stt_server/whisper_online_server.py"
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    import logging
    import sys

    namespace = dict(np=np, json=json, SAMPLING_RATE=16000,
                     logger=logging.getLogger(__name__), sys=sys)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ServerProcessor"]


def test_empty_vad_final_is_sent():
    """Previously confirmed words must receive an utterance boundary."""
    lines = []
    processor = server_class()(SimpleNamespace(send=lines.append), None, 0.04)
    processor.send_result((0, 1, "Roll initiative"), False)
    processor.send_result((None, None, ""), True)
    assert len(lines) == 2
    assert json.loads(lines[-1]) == {"text": "", "end": 1, "is_final": True}


def test_tcp_fragmentation_preserves_samples():
    """Odd TCP packet boundaries must not corrupt PCM samples."""
    pcm = np.arange(640, dtype='<i2').tobytes()
    packets = [pcm[:1], pcm[1:5], pcm[5:]]

    def receive(size):
        return packets.pop(0) if packets else b''

    processor = server_class()(SimpleNamespace(non_blocking_receive_audio=receive), None, 0.04)
    np.testing.assert_allclose(processor.receive_audio_chunk(), np.arange(640) / 32768)


def test_eof_decodes_tail_before_finalizing():
    """Speech since the last decoder iteration survives stopping capture."""
    calls, lines = [], []

    def decode():
        calls.append('decode')
        return (0, 1, 'last words')

    def finish():
        calls.append('finish')
        return (None, None, '')

    online = SimpleNamespace(online=SimpleNamespace(audio_buffer=[1], process_iter=decode),
                             finish=finish)
    processor = server_class()(SimpleNamespace(send=lines.append), online, 0.04)
    processor.flush_final()
    assert calls == ['decode', 'finish']
    assert json.loads(lines[-1])['is_final'] is True


def test_eof_after_vad_final_does_not_repeat_utterance():
    """Disconnecting after silence must not decode the previous speech again."""
    lines = []
    processor = server_class()(SimpleNamespace(send=lines.append),
                               SimpleNamespace(status='nonvoice'), 0.04)
    processor.flush_final()
    assert lines == []
