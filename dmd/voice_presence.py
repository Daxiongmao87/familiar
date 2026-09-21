"""Persistent voice presence for Familiar: follows the DM's voice channel."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class VoicePresence:
    """Keeps Familiar in the DM's voice channel and feeds SpeakingTracker."""

    def __init__(
        self, token: str, guild_id: int, dm_user_id: int | None, tracker: Any
    ) -> None:
        self._token = token
        self._guild_id = int(guild_id)
        self._dm_user_id = int(dm_user_id) if dm_user_id is not None else None
        self._tracker = tracker
        self._client: Any = None
        self._voice: Any = None
        self._task: asyncio.Task | None = None
        self._follow_lock = asyncio.Lock()

    async def start(self) -> None:
        import discord

        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            logger.info("voice presence ready as %s", self._client.user)
            await self._follow_dm()

        @self._client.event
        async def on_voice_state_update(member: Any, before: Any, after: Any) -> None:
            if getattr(member, "id", None) == self._dm_user_id:
                await self._follow_dm()
            # speaking state is via member_speaking_state_update, not voice_state

        @self._client.event
        async def on_member_speaking_state_update(
            member: Any, ssrc: Any, state: Any
        ) -> None:
            try:
                speaking = (
                    bool(getattr(state, "value", 0) & 1)
                    if hasattr(state, "value")
                    else bool(state)
                )
                uid = getattr(member, "id", None)
                if uid is None:
                    guild = self._client.get_guild(self._guild_id)
                    voice = getattr(guild, "voice_client", None) or self._voice
                    uid = getattr(voice, "_ssrc_to_id", {}).get(ssrc)
                    if uid is None:
                        logger.warning("unresolved Discord speaking stream ssrc=%s", ssrc)
                        return
                    member = guild.get_member(uid) if guild is not None else None
                name = str(
                    getattr(member, "display_name", "")
                    or getattr(member, "global_name", "")
                    or ""
                )
                self._tracker.on_speaking(
                    str(uid), speaking, name=name if name else None
                )
            except Exception:
                pass

        self._task = asyncio.create_task(self._client.start(self._token))

    async def _follow_dm(self) -> None:
        """Join the DM's channel, recovering stale Discord voice clients."""
        async with self._follow_lock:
            await self._follow_dm_locked()

    async def _discard_voice(self, voice: Any) -> None:
        """Force-disconnect and unregister a stale Discord voice client."""
        try:
            await asyncio.wait_for(voice.disconnect(force=True), timeout=5.0)
        except Exception as exc:
            logger.warning("stale voice disconnect failed: %s", exc)
            cleanup = getattr(voice, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        finally:
            if self._voice is voice:
                self._voice = None

    async def _follow_dm_locked(self) -> None:
        """Serialized implementation of the DM-follow state transition."""
        if self._client is None or self._dm_user_id is None:
            return
        guild = self._client.get_guild(self._guild_id)
        if guild is None:
            return
        vs = getattr(guild, "_voice_states", {}) or {}
        dm_vs = vs.get(int(self._dm_user_id))
        dm_channel = getattr(dm_vs, "channel", None) if dm_vs is not None else None
        # discord.py can retain a guild-registered VoiceClient after a 4006
        # disconnect even though ``is_connected()`` is false. Reusing only
        # ``self._voice`` then makes channel.connect() raise "Already
        # connected" forever. Treat the guild registry as authoritative.
        voice = self._voice or getattr(guild, "voice_client", None)
        if dm_channel is None:
            # DM not in voice — leave if we are in one
            if voice is not None:
                await self._discard_voice(voice)
            return
        # already in correct channel?
        if (
            voice
            and getattr(voice, "channel", None)
            and voice.channel.id == dm_channel.id
            and voice.is_connected()
        ):
            self._voice = voice
            return
        # need to move or join
        try:
            if voice and voice.is_connected():
                await voice.move_to(dm_channel)
                self._voice = voice
                logger.info("Familiar moved to %s for DM", dm_channel.name)
            else:
                if voice is not None:
                    await self._discard_voice(voice)
                self._voice = await dm_channel.connect()
                logger.info("Familiar joined %s for DM", dm_channel.name)
        except Exception as exc:
            logger.warning("follow DM failed: %s", exc)

    async def stop(self, timeout_s: float = 10.0) -> None:
        """Tear down voice presence, bounded so a stuck Discord login can't
        wedge the process on shutdown.

        If any cleanup await hangs past ``timeout_s``, log and abandon: the
        event loop will reap the abandoned task on close.
        """

        async def _teardown() -> None:
            if self._voice and self._voice.is_connected():
                try:
                    await self._voice.disconnect(force=True)
                except Exception:
                    pass
            if self._client and not self._client.is_closed():
                try:
                    await self._client.close()
                except Exception:
                    pass
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass

        try:
            await asyncio.wait_for(_teardown(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "voice presence stop timed out after %.1fs; abandoning", timeout_s
            )
