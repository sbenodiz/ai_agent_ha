"""Regression tests for issue #82: AI Agent HA corrupts automations.yaml.

Before the fix, ``create_automation`` wrote the whole file back with a plain
``yaml.dump(..., default_flow_style=False)`` to an open file handle. That:

  * escaped every non-ASCII character into ``\\uXXXX`` ("heavily escaped
    strings" in the bug report),
  * rewrote/reordered the user's *existing* automations,
  * left no backup, and
  * could leave a half-written file (handle never explicitly closed/flushed
    before ``automation.reload`` ran).

It also silently dropped a single-mapping ``trigger``/``action`` (only lists
were kept), which then raised ``KeyError`` for otherwise-valid automations.

These tests pin the fixed behavior:
  1. non-ASCII text is preserved verbatim, not escaped;
  2. pre-existing automations survive a write byte-for-byte in meaning, with
     key order preserved;
  3. a ``.bak`` backup is created before overwriting;
  4. the write is atomic (no leftover temp files; existing file untouched on a
     failed write);
  5. a single-mapping trigger/action is normalized, not dropped.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

# Add the repo root to the path for direct imports (mirrors test_issue_80.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_agent_with_tmp_automations(tmp_path, initial_yaml=None):
    """Build an agent whose hass points automations.yaml at a temp file.

    Returns (agent, automations_path). ``async_add_executor_job`` is wired to
    actually run the target callable so the real read/write helpers execute,
    and ``services.async_call`` is a no-op coroutine.
    """
    from custom_components.ai_agent_ha.agent import AiAgentHaAgent

    automations_path = os.path.join(tmp_path, "automations.yaml")
    if initial_yaml is not None:
        with open(automations_path, "w", encoding="utf-8") as handle:
            handle.write(initial_yaml)

    hass = MagicMock()
    hass.data = {"ai_agent_ha": {"configs": {}}}
    hass.config.path = lambda *parts: os.path.join(tmp_path, *parts)

    async def _run_executor_job(func, *args):
        return func(*args)

    async def _async_call(*args, **kwargs):
        return None

    hass.async_add_executor_job = _run_executor_job
    hass.services.async_call = _async_call

    config = {
        "ai_provider": "anthropic",
        "anthropic_token": "sk-ant-" + "x" * 40,
        "models": {"anthropic": "claude-sonnet-4-5-20250929"},
    }
    agent = AiAgentHaAgent(hass, config)
    return agent, automations_path


# A realistic, hand-written automations.yaml as produced by the HA UI editor:
# non-ASCII alias/description (the reporter is Roman Farkaš -> Czech) plus a
# Jinja template in an action.
EXISTING_YAML = """\
- id: '1700000000000'
  alias: Osvětlení ložnice ráno
  description: Ráno rozsvítí světlo v ložnici když je tma
  trigger:
    - platform: time
      at: '06:30:00'
  condition: []
  action:
    - service: light.turn_on
      target:
        entity_id: light.bedroom
      data:
        message: "Teplota je {{ states('sensor.temp') }}°C"
  mode: single
"""


class TestUnicodePreserved:
    """Fix 1: non-ASCII text is never escaped to \\uXXXX."""

    @pytest.mark.asyncio
    async def test_accented_text_written_verbatim(self):
        try:
            from custom_components.ai_agent_ha.agent import (  # noqa: F401
                AiAgentHaAgent,
            )
        except ImportError:
            pytest.skip("agent module not available")

        with tempfile.TemporaryDirectory() as tmp:
            agent, path = _make_agent_with_tmp_automations(tmp)
            result = await agent.create_automation(
                {
                    "alias": "Zhasni světla ve 22:00",
                    "description": "Každý den ve 22:00 zhasne všechna světla",
                    "trigger": [{"platform": "time", "at": "22:00:00"}],
                    "action": [
                        {"service": "light.turn_off", "target": {"entity_id": "all"}}
                    ],
                }
            )
            assert result.get("success") is True, result

            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()

        # The accented text must appear literally...
        assert "Zhasni světla ve 22:00" in raw
        assert "Každý den ve 22:00 zhasne všechna světla" in raw
        # ...and there must be no escape sequences at all.
        assert "\\u" not in raw
        assert "\\x" not in raw


class TestExistingAutomationsPreserved:
    """Fix 2: pre-existing automations survive intact, key order preserved."""

    @pytest.mark.asyncio
    async def test_existing_automation_not_corrupted(self):
        try:
            import yaml  # noqa: F401

            from custom_components.ai_agent_ha.agent import (  # noqa: F401
                AiAgentHaAgent,
            )
        except ImportError:
            pytest.skip("agent module not available")
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            agent, path = _make_agent_with_tmp_automations(tmp, EXISTING_YAML)
            result = await agent.create_automation(
                {
                    "alias": "New one",
                    "trigger": [{"platform": "state", "entity_id": "binary_sensor.x"}],
                    "action": [{"service": "light.turn_on"}],
                }
            )
            assert result.get("success") is True, result

            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            reparsed = yaml.safe_load(raw)

        original = yaml.safe_load(EXISTING_YAML)[0]
        # The original automation is still present and byte-for-byte equal in
        # meaning (same dict), and its accented text is still readable on disk.
        assert reparsed[0] == original
        assert "Osvětlení ložnice ráno" in raw
        # Key order preserved: id comes before alias (sort_keys=False).
        assert raw.index("id:") < raw.index("alias:")
        # The new automation was appended, not prepended.
        assert reparsed[1]["alias"] == "New one"


class TestBackupCreated:
    """Fix 3: a .bak is written before overwriting an existing file."""

    @pytest.mark.asyncio
    async def test_backup_file_created(self):
        try:
            from custom_components.ai_agent_ha.agent import (  # noqa: F401
                AiAgentHaAgent,
            )
        except ImportError:
            pytest.skip("agent module not available")

        with tempfile.TemporaryDirectory() as tmp:
            agent, path = _make_agent_with_tmp_automations(tmp, EXISTING_YAML)
            await agent.create_automation(
                {
                    "alias": "Another",
                    "trigger": [{"platform": "state", "entity_id": "x.y"}],
                    "action": [{"service": "light.turn_on"}],
                }
            )
            backup = path + ".bak"
            assert os.path.exists(backup), "expected automations.yaml.bak"
            with open(backup, "r", encoding="utf-8") as handle:
                assert handle.read() == EXISTING_YAML


class TestAtomicWrite:
    """Fix 4: write is atomic; no temp file leaks; bad write keeps original."""

    @pytest.mark.asyncio
    async def test_no_temp_file_left_behind(self):
        try:
            from custom_components.ai_agent_ha.agent import (  # noqa: F401
                AiAgentHaAgent,
            )
        except ImportError:
            pytest.skip("agent module not available")

        with tempfile.TemporaryDirectory() as tmp:
            agent, path = _make_agent_with_tmp_automations(tmp, EXISTING_YAML)
            await agent.create_automation(
                {
                    "alias": "Yet another",
                    "trigger": [{"platform": "state", "entity_id": "x.y"}],
                    "action": [{"service": "light.turn_on"}],
                }
            )
            leftovers = [
                name
                for name in os.listdir(tmp)
                if name.endswith(".tmp") or name.startswith(".automations.")
            ]
            assert leftovers == [], f"temp files left behind: {leftovers}"

    def test_write_helper_leaves_original_on_failure(self):
        """A YAML-serialization failure must not destroy the existing file."""
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent
        except ImportError:
            pytest.skip("agent module not available")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "automations.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(EXISTING_YAML)

            # An object PyYAML cannot represent forces yaml.dump to raise; the
            # original file must be untouched and no temp file left behind.
            class Unserializable:
                pass

            with pytest.raises(Exception):
                AiAgentHaAgent._write_automations_file(
                    path, [{"alias": "boom", "x": Unserializable()}]
                )

            with open(path, "r", encoding="utf-8") as handle:
                assert handle.read() == EXISTING_YAML
            leftovers = [n for n in os.listdir(tmp) if n != "automations.yaml"]
            assert leftovers == [], f"unexpected leftovers: {leftovers}"

    def test_original_permission_bits_preserved(self):
        """Atomic replace must keep the file's mode, not reset it to 0600."""
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent
        except ImportError:
            pytest.skip("agent module not available")

        import stat

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "automations.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(EXISTING_YAML)
            # Group/world-readable, like a file edited over a share.
            os.chmod(path, 0o664)

            AiAgentHaAgent._write_automations_file(
                path,
                [{"alias": "x", "trigger": [{"platform": "time"}], "action": []}],
            )

            mode = stat.S_IMODE(os.stat(path).st_mode)
            assert mode == 0o664, f"permission bits not preserved: {oct(mode)}"


class TestSanitizeNormalizesSingleMapping:
    """Fix 5: a single-mapping trigger/action is normalized, not dropped."""

    def test_single_mapping_trigger_action_kept(self):
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent
        except ImportError:
            pytest.skip("agent module not available")

        hass = MagicMock()
        hass.data = {"ai_agent_ha": {"configs": {}}}
        config = {
            "ai_provider": "anthropic",
            "anthropic_token": "sk-ant-" + "x" * 40,
            "models": {"anthropic": "claude-sonnet-4-5-20250929"},
        }
        agent = AiAgentHaAgent(hass, config)

        sanitized = agent._sanitize_automation_config(
            {
                "alias": "Single",
                "trigger": {"platform": "time", "at": "07:00:00"},
                "action": {"service": "light.turn_on"},
            }
        )
        assert sanitized["trigger"] == [{"platform": "time", "at": "07:00:00"}]
        assert sanitized["action"] == [{"service": "light.turn_on"}]

    @pytest.mark.asyncio
    async def test_single_mapping_automation_created(self):
        try:
            from custom_components.ai_agent_ha.agent import (  # noqa: F401
                AiAgentHaAgent,
            )
        except ImportError:
            pytest.skip("agent module not available")

        with tempfile.TemporaryDirectory() as tmp:
            agent, path = _make_agent_with_tmp_automations(tmp)
            result = await agent.create_automation(
                {
                    "alias": "Single mapping",
                    "trigger": {"platform": "time", "at": "07:00:00"},
                    "action": {"service": "light.turn_on"},
                }
            )
            assert result.get("success") is True, result


class TestMissingPiecesRejectedCleanly:
    """A malformed trigger/action is rejected with a clear error, not KeyError."""

    @pytest.mark.asyncio
    async def test_missing_action_returns_clear_error(self):
        try:
            from custom_components.ai_agent_ha.agent import (  # noqa: F401
                AiAgentHaAgent,
            )
        except ImportError:
            pytest.skip("agent module not available")

        with tempfile.TemporaryDirectory() as tmp:
            agent, _ = _make_agent_with_tmp_automations(tmp)
            # action is a bare string -> dropped by the sanitizer
            result = await agent.create_automation(
                {
                    "alias": "Bad",
                    "trigger": [{"platform": "time", "at": "07:00:00"}],
                    "action": "not-a-valid-action",
                }
            )
            assert result.get("success") is not True
            assert "action" in result.get("error", "").lower()
