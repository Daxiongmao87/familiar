#!/usr/bin/env python3
# Familiar vendored STT server.
#
# Upstream: ufal/whisper_streaming @ 6da90b44b7e50d79695e68166d2a2c7609c75abb
# (whisper_online_server.py, 6097 bytes pristine). Verify:
#   curl -s https://raw.githubusercontent.com/ufal/whisper_streaming/\
#   6da90b44b7e50d79695e68166d2a2c7609c75abb/whisper_online_server.py | sha256sum
#   pristine sha256: d89178d8a57c646ab46f76e9d4c6957e82c6e75fffb75438516ad81983101101
#
# FAMILIAR PATCH (marked inline): stock upstream sends plain-text
# "beg_ms end_ms text" lines with no utterance-final signal, which
# dmd/streaming_stt.py cannot use (it needs VAD finals to commit
# transcripts). This file instead emits newline-delimited JSON
# {"text","start","end","is_final"} per confirmed increment, where
# is_final is sampled from the VAC processor's is_currently_final flag
# before process_iter() consumes it. MUST run with --vac: without it
# every line is a partial and no transcript ever commits.
from whisper_online import *

import sys
import argparse
import json
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args,logger,other="")

# setting whisper object by args

SAMPLING_RATE = 16000

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# warm up the ASR because the very first transcribe takes more time than the others.
# Test results in https://github.com/ufal/whisper_streaming/pull/81
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file,0,1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. "+msg)
        sys.exit(1)
else:
    logger.warning(msg)


######### Server objects

import line_packet
import socket

class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000*5*60 # 5 minutes # was: 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self, size):
        try:
            r = self.conn.recv(size)
            return r
        except ConnectionResetError:
            return None


import io
import soundfile

# wraps socket and ASR object, and serves one client connection.
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

        self.is_first = True

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = bytearray()
        minlimit = max(1, int(self.min_chunk * SAMPLING_RATE)) * 2
        while len(out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio(minlimit - len(out))
            if not raw_bytes:
                break
            out.extend(raw_bytes)
        if not out:
            return None
        # TCP may split a sample across packets; decode only after joining.
        return np.frombuffer(out[:len(out) - len(out) % 2], dtype='<i2').astype(np.float32) / 32768

    # FAMILIAR PATCH: JSON wire format with VAD finals (stock sent plain
    # "beg_ms end_ms text" with no final signal). start/end stay in
    # seconds of stream audio, matching dmd/streaming_stt.py. The stderr
    # log keeps the stock human-readable shape.
    def send_result(self, o, is_final):
        if o[0] is None:
            if is_final:
                self.connection.send(json.dumps({
                    "text": "", "end": self.last_end, "is_final": True,
                }))
            logger.debug("No text in this segment")
            return
        beg, end = o[0], o[1]
        if self.last_end is not None:
            beg = max(beg, self.last_end)
        self.last_end = end
        print("%1.0f %1.0f %s" % (beg * 1000, end * 1000, o[2]),
              flush=True, file=sys.stderr)
        msg = json.dumps(
            {"text": o[2], "start": beg, "end": end, "is_final": is_final}
        )
        self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                self.flush_final()
                break
            self.online_asr_proc.insert_audio_chunk(a)
            # FAMILIAR PATCH: VAC sets is_currently_final during
            # insert_audio_chunk when VAD closes an utterance, and
            # process_iter()/finish() consumes it — sample it first.
            # Without --vac the flag never exists: every line stays a
            # partial (documented at the top of this file).
            pending_final = bool(
                getattr(self.online_asr_proc, 'is_currently_final', False)
            )
            if pending_final:
                # Decode the last audio since the previous online update
                # before finish() flushes its hypothesis buffer.
                self.send_result(self.online_asr_proc.online.process_iter(), False)
            o = self.online_asr_proc.process_iter()
            try:
                self.send_result(o, pending_final)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

    def flush_final(self):
        """Flush remaining speech when the client half-closes its PCM stream."""
        if getattr(self.online_asr_proc, 'status', None) == 'nonvoice':
            return  # VAD already committed this utterance; do not decode it again.
        processor = getattr(self.online_asr_proc, 'online', self.online_asr_proc)
        try:
            if len(processor.audio_buffer):
                self.send_result(processor.process_iter(), False)
            self.send_result(self.online_asr_proc.finish(), True)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("client disconnected before final flush")



# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    logger.info('Listening on'+str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        logger.info('Connected to client on {}'.format(addr))
        connection = Connection(conn)
        proc = ServerProcessor(connection, online,
                               args.vac_chunk_size if args.vac else args.min_chunk_size)
        proc.process()
        conn.close()
        logger.info('Connection to client closed')
logger.info('Connection closed, terminating.')
