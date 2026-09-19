"""Regression: create_app must deliver externally-published events to /ws.

Guards the two-EventBus bug where engine/pool published to an external bus
while /ws subscribed to a private internal one.
"""

import time

from fastapi.testclient import TestClient

from dmd.server import EventBus, create_app


def test_injected_bus_delivers_external_events_to_ws() -> None:
    bus = EventBus()
    app = create_app(cfg=None, engine=None, init_runner=None, bus=bus)
    client = TestClient(app)
    with client:
        with client.websocket_connect("/ws") as ws:
            bus.publish_sync(
                {"type": "status", "state": "regression", "detail": "injected-bus", "t": time.time()}
            )
            msg = ws.receive_json()
    assert msg["type"] == "status"
    assert msg["detail"] == "injected-bus"
