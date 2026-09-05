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
                self._tracker.on_speaking(str(getattr(member, "id", ssrc)), speaking)
            except Exception:
                pass

        self._task = asyncio.create_task(self._client.start(self._token))

    async def _follow_dm(self) -> None:
        if self._client is None or self._dm_user_id is None:
            return
        guild = self._client.get_guild(self._guild_id)
        if guild is None:
            return
        vs = getattr(guild, "_voice_states", {}) or {}
        dm_vs = vs.get(int(self._dm_user_id))
        dm_channel = getattr(dm_vs, "channel", None) if dm_vs is not None else None
        if dm_channel is None:
            # DM not in voice — leave if we are in one
            if self._voice and self._voice.is_connected():
                try:
                    await self._voice.disconnect(force=True)
                except Exception:
                    pass
                self._voice = None
            return
        # already in correct channel?
        if (
            self._voice
            and getattr(self._voice, "channel", None)
            and self._voice.channel.id == dm_channel.id
            and self._voice.is_connected()
        ):
            return
        # need to move or join
        try:
            if self._voice and self._voice.is_connected():
                await self._voice.move_to(dm_channel)
                logger.info("Familiar moved to %s for DM", dm_channel.name)
            else:
                # fresh join via channel.connect

                # use guild's voice channel connect helper
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
