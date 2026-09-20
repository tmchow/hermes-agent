"""Inbound dispatch + dedup tests for PhotonAdapter.

These exercise the sidecar-event stream and parsing without spawning the
Node sidecar or binding ports.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.profile_routing import parse_profile_routes
from gateway.session_identity import identity_of
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture(adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _dm_event(text: str, msg_id: str = "spc-msg-abc") -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "platform": "iMessage",
        "space": {"id": "+15551234567", "type": "dm", "phone": "+15551234567"},
        "sender": {"id": "+15551234567"},
        "content": {"type": "text", "text": text},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


@pytest.mark.asyncio
async def test_dispatch_text_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event("hello world"))

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "hello world"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "spc-msg-abc"
    src = event.source
    assert src is not None
    assert src.platform == Platform("photon")
    assert src.chat_id == "+15551234567"
    assert src.chat_type == "dm"
    assert src.user_id == "+15551234567"


@pytest.mark.asyncio
async def test_dispatch_read_receipt_does_not_wake_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    receipt = _dm_event("", msg_id="spc-read-1")
    receipt["content"] = {
        "type": "read",
        "targetMessageId": "bot-msg-1",
        "targetDirection": "outbound",
    }

    await adapter._dispatch_inbound(receipt)

    assert captured == []


@pytest.mark.asyncio
async def test_dispatch_read_receipt_alias_does_not_wake_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some spectrum-ts streams label receipts ``read_receipt`` — same drop."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    receipt = _dm_event("", msg_id="spc-read-2")
    receipt["content"] = {
        "type": "read_receipt",
        "targetMessageId": "bot-msg-2",
        "targetDirection": "outbound",
    }

    await adapter._dispatch_inbound(receipt)

    assert captured == []


# A real 1x1 transparent PNG (passes base.py's _looks_like_image magic check).
_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhf"
    "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _attachment_event(
    content: Dict[str, Any], msg_id: str = "spc-msg-att", chat_id: str = "+155****4567"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "space": {"id": chat_id, "type": "dm", "phone": chat_id},
        "sender": {"id": chat_id},
        "content": {"type": "attachment", **content},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _configure_multiplex_runner(
    adapter: PhotonAdapter,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    route_profile: str = "secondary-profile",
    serve_route_profile: bool = True,
) -> None:
    """Attach the real current-main identity resolver to a Photon adapter."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform("photon"): adapter.config}
    runner.config.profile_routes = parse_profile_routes(
        [{
            "name": "photon-secondary",
            "platform": "photon",
            "profile": route_profile,
            "chat_id": "secondary-chat",
        }]
    )
    runner._primary_profile_name = "default"
    runner.adapters = {Platform("photon"): adapter}
    runner._profile_adapters = {route_profile: {}} if serve_route_profile else {}
    monkeypatch.setattr(adapter, "gateway_runner", runner)

    served_names = {"default"}
    if serve_route_profile:
        served_names.add(route_profile)
    served = [
        (name, root if name == "default" else root / "profiles" / name)
        for name in sorted(served_names)
    ]
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex=False: served,
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: root if name == "default" else root / "profiles" / name,
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda name: name in served_names,
    )


def _voice_event(
    content: Dict[str, Any], msg_id: str = "spc-msg-voice"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "space": {"id": "+15551234567", "type": "dm", "phone": "+15551234567"},
        "sender": {"id": "+15551234567"},
        "content": {"type": "voice", **content},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "mime", "raw"),
    [
        (
            "score.png",
            "image/png",
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYA"
                "AjCB0C8AAAAASUVORK5CYII="
            ),
        ),
        (
            "score.heic",
            "image/heic",
            b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 32,
        ),
    ],
)
async def test_routed_attachment_cached_under_destination_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mime: str,
    raw: bytes,
) -> None:
    """A shared Photon listener must cache bytes under the routed profile.

    The source profile is known before normalization. Caching under the
    gateway's launch/default home leaves a Docker-isolated child profile with
    a path its image resolver must reject, even though the attachment belongs
    to that profile's current conversation.
    """
    root = tmp_path / "hermes"
    secondary_home = root / "profiles" / "secondary-profile"
    secondary_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    adapter = _make_adapter(monkeypatch)
    _configure_multiplex_runner(adapter, root, monkeypatch)
    captured = _capture(adapter, monkeypatch)

    async def send_default(label: str) -> None:
        payload = f"default-{label}".encode()
        await adapter._dispatch_inbound(
            _attachment_event(
                {
                    "name": f"{label}.txt",
                    "mimeType": "text/plain",
                    "size": len(payload),
                    "data": base64.b64encode(payload).decode("ascii"),
                    "encoding": "base64",
                },
                msg_id=f"default-{label}",
                chat_id=f"default-{label}",
            )
        )

    await send_default("before")
    event = _attachment_event(
        {
            "name": name,
            "mimeType": mime,
            "size": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        },
        chat_id="secondary-chat",
    )

    await adapter._dispatch_inbound(event)
    await send_default("after")

    assert len(captured) == 3
    default_before, routed, default_after = captured
    identities = [identity_of(item.source) for item in captured]
    assert all(identity is not None for identity in identities)
    assert [identity.runtime_profile for identity in identities if identity is not None] == [
        "default",
        "secondary-profile",
        "default",
    ]
    for item in (default_before, default_after):
        assert Path(item.media_urls[0]).resolve().is_relative_to((root / "cache").resolve())

    identity = identities[1]
    assert identity is not None
    assert (identity.transport_profile, identity.runtime_profile) == (
        "default",
        "secondary-profile",
    )
    assert identity.runtime_home == secondary_home
    assert len(routed.media_urls) == 1
    cached = Path(routed.media_urls[0]).resolve()
    assert cached.is_relative_to(secondary_home.resolve())
    assert cached.read_bytes() == raw
    assert not (root / "cache" / "images" / cached.name).exists()

    # A Docker-isolated secondary can inspect and edit the current attachment
    # without receiving general host-file access or an active sandbox session.
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.image_source import ResolveContext, resolve_image_source, resolve_local_source_to_data_url

    token = set_hermes_home_override(str(secondary_home))
    try:
        resolved = await resolve_image_source(str(cached), ResolveContext(task_id="never-started"))
        edit_source = await resolve_local_source_to_data_url(str(cached), task_id="never-started")
    finally:
        reset_hermes_home_override(token)
    assert resolved.data == raw
    assert base64.b64decode(edit_source.partition(",")[2]) == raw


@pytest.mark.asyncio
async def test_unserved_routed_attachment_is_dropped_before_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale route cannot persist bytes or recreate its deleted profile."""
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))

    adapter = _make_adapter(monkeypatch)
    _configure_multiplex_runner(
        adapter,
        root,
        monkeypatch,
        route_profile="deleted-profile",
        serve_route_profile=False,
    )
    captured = _capture(adapter, monkeypatch)
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYA"
        "AjCB0C8AAAAASUVORK5CYII="
    )

    await adapter._dispatch_inbound(
        _attachment_event(
            {
                "name": "late.png",
                "mimeType": "image/png",
                "size": len(raw),
                "data": base64.b64encode(raw).decode("ascii"),
                "encoding": "base64",
            },
            chat_id="secondary-chat",
        )
    )

    assert captured == []
    assert not (root / "cache").exists()
    assert not (root / "profiles" / "deleted-profile").exists()


@pytest.mark.asyncio
async def test_on_inbound_line_dispatches_and_dedups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    line = json.dumps(_dm_event("ping", msg_id="dup-1"))
    await adapter._on_inbound_line(line)
    await adapter._on_inbound_line(line)  # same messageId -> deduped

    assert len(captured) == 1
    assert captured[0].text == "ping"


@pytest.mark.asyncio
async def test_ndjson_stream_preserves_unicode_line_separators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    separated = "first\u2028second\u2029third\u0085fourth"
    payloads = [
        json.dumps(_dm_event(separated, msg_id="unicode-lines"), ensure_ascii=False),
        json.dumps(_dm_event("ordinary", msg_id="ordinary-line")),
    ]
    stream = payloads[0] + "\n\n" + payloads[1]
    received: List[str] = []

    class ChunkedResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_text(self):
            for chunk in (stream[:17], stream[17:43], stream[43:]):
                yield chunk

        async def aiter_lines(self):
            for line in stream.splitlines():
                yield line

    class Client:
        def stream(self, *_args, **_kwargs):
            return ChunkedResponse()

    async def capture_line(line: str) -> None:
        received.append(line)
        if len(received) == 2:
            adapter._inbound_running = False

    adapter._http_client = Client()
    adapter._inbound_running = True
    monkeypatch.setattr(adapter, "_on_inbound_line", capture_line)

    await adapter._inbound_loop()

    assert received == payloads


def test_is_duplicate_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter._dedup.is_duplicate("id-1") is False
    assert adapter._dedup.is_duplicate("id-1") is True
    assert adapter._dedup.is_duplicate("id-2") is False
    assert adapter._dedup.is_duplicate("id-1") is True  # still dup


def test_check_requirements_without_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # If no node binary on PATH the adapter should refuse to start.
    from plugins.platforms.photon import adapter as adapter_mod

    monkeypatch.setattr(adapter_mod.shutil, "which", lambda _name: None)
    assert adapter_mod.check_requirements() is False


# ---------------------------------------------------------------------------
# CAF attachment promotion + U+FFFC placeholder tests
# ---------------------------------------------------------------------------

_CAF_BYTES = b"caff" + b"\x00" * 60  # Minimal CAF header magic


def _caf_attachment_event(
    content: Dict[str, Any], msg_id: str = "spc-msg-caf"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "space": {"id": "+155****4567", "type": "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": {"type": "attachment", **content},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


@pytest.mark.asyncio
async def test_caf_attachment_named_promoted_to_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named .caf attachment is promoted to VOICE for STT routing."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    raw = _CAF_BYTES
    event = _caf_attachment_event(
        {
            "name": "voice_note.caf",
            "mimeType": "audio/x-caf",
            "size": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.VOICE
    assert ev.media_types == ["audio/x-caf"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.read_bytes() == raw
        assert ev.text == "(voice)"
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_fffc_placeholder_no_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A U+FFFC placeholder text does not trigger a message dispatch."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _dm_event("\ufffc", msg_id="spc-msg-fffc")
    chat_key = event["space"]["id"]
    await adapter._dispatch_inbound(event)

    assert len(captured) == 0
    assert chat_key in adapter._pending_fffc


@pytest.mark.asyncio
async def test_disconnect_cancels_pending_fffc_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """disconnect() cancels any pending U+FFFC placeholder tasks."""
    adapter = _make_adapter(monkeypatch)
    _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event("\ufffc", msg_id="spc-msg-fffc"))
    assert len(adapter._pending_fffc) == 1

    async def _noop_stop_sidecar():
        pass

    monkeypatch.setattr(adapter, "_stop_sidecar", _noop_stop_sidecar)
    monkeypatch.setattr(adapter, "_inbound_running", False)
    monkeypatch.setattr(adapter, "_inbound_task", None)
    monkeypatch.setattr(adapter, "_sidecar_health_task", None)
    monkeypatch.setattr(adapter, "_http_client", None)

    await adapter.disconnect()

    assert len(adapter._pending_fffc) == 0
