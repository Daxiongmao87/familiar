"""Voice-presence state transition regression tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dmd.voice_presence import VoicePresence
import asyncio
import sys
import pytest
from dmd.speaking_tracker import SpeakingTracker


@pytest.mark.parametrize("mapping,expected", [({1502: 188660400722673664}, "188660400722673664"), ({}, None)])
async def test_missing_member_never_becomes_stream_identity(monkeypatch, mapping, expected):
    """Exercise the registered Discord callback with an uncached member."""
    class Client:
        def __init__(self, **kwargs):
            self.guild = SimpleNamespace(voice_client=SimpleNamespace(_ssrc_to_id=mapping),
                                         get_member=lambda uid: None)
        def event(self, callback):
            setattr(self, callback.__name__, callback)
            return callback
        def get_guild(self, uid):
            return self.guild
        async def start(self, token):
            return None
    monkeypatch.setitem(sys.modules, "discord", SimpleNamespace(
        Client=Client, Intents=SimpleNamespace(none=lambda: SimpleNamespace())))
    tracker = SpeakingTracker()
    presence = VoicePresence("test", 10, 20, tracker)
    await presence.start()
    await presence._client.on_member_speaking_state_update(None, 1502, 1)
    assert set(tracker.snapshot()["active"]) == ({expected} if expected else set())
    await presence._client.on_member_speaking_state_update(None, 1502, 0)
    assert tracker.snapshot()["active"] == {}
    await presence._task


class _Voice:
    def __init__(self, channel: Any, connected: bool) -> None:
        self.channel = channel
        self.connected = connected
        self.disconnected = False
        self.cleaned = False

    def is_connected(self) -> bool:
        return self.connected

    async def disconnect(self, *, force: bool) -> None:
        assert force is True
        self.disconnected = True
        self.connected = False

    def cleanup(self) -> None:
        self.cleaned = True


class _Channel:
    def __init__(self) -> None:
        self.id = 30
        self.name = "table"
        self.stale: _Voice | None = None
        self.fresh = _Voice(self, True)
        self.connect_calls = 0

    async def connect(self) -> _Voice:
        assert self.stale is None or self.stale.disconnected
        self.connect_calls += 1
        return self.fresh


async def test_follow_recovers_disconnected_guild_registered_voice() -> None:
    """A Discord 4006 ghost client must not block every later join."""
    channel = _Channel()
    stale = _Voice(channel, False)
    channel.stale = stale
    guild = SimpleNamespace(
        _voice_states={20: SimpleNamespace(channel=channel)},
        voice_client=stale,
    )
    presence = VoicePresence("token", guild_id=10, dm_user_id=20, tracker=object())
    presence._client = SimpleNamespace(get_guild=lambda guild_id: guild)
    presence._voice = stale

    await presence._follow_dm()

    assert stale.disconnected is True
    assert channel.connect_calls == 1
    assert presence._voice is channel.fresh


async def test_follow_keeps_healthy_connection_in_target_channel() -> None:
    """The recovery path must not churn a healthy voice connection."""
    channel = _Channel()
    voice = _Voice(channel, True)
    guild = SimpleNamespace(
        _voice_states={20: SimpleNamespace(channel=channel)},
        voice_client=voice,
    )
    presence = VoicePresence("token", guild_id=10, dm_user_id=20, tracker=object())
    presence._client = SimpleNamespace(get_guild=lambda guild_id: guild)
    presence._voice = voice

    await presence._follow_dm()

    assert voice.disconnected is False
    assert channel.connect_calls == 0
    assert presence._voice is voice
