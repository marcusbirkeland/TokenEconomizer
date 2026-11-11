"""The DSH host, end to end.

DSH's bundle (dsh-config) speaks the dispatcher's protocol on DSH's behalf, exactly as the Pi
extension does: one `te-hook --host dsh` call per event, and the session's export as a
Claude-format JSONL (the bundle's adapter writes it from DSH's zstd session log). So the
mechanisms see the same shapes they see under Claude Code, and the DSH-specific surface is thin:

  * the host is registered and detected (`--host dsh`, or `TE_DSH=1` in the hook's environment);
  * a whole repository life -- sessions starting, ending, folded -- runs on that host, with the
    fold reading the Claude-format export the bundle sends (`session_usage` / `prompt_count` /
    the base `tool_usage` all read it);
  * the side call (`dsh headless --json`, prompt on stdin) is parsed from its NDJSON run events
    and charged in the ledger's fields, and a run that says the provider is unsigned in pauses
    the host's side calls like any other host's.
"""
import io
import json
import os
import stat
import textwrap

import pytest

from tokeneconomizer.plugin import hook, hosts, sidecall
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
from tokeneconomizer.plugin.store import Store

from fixtures import write_claude_transcript


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0",
            "TE_MICROSKILLS": "envfacts",
            "TE_DSH": "1", "TE_DSH_MODE": "tty",
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    (r / "pyproject.toml").write_text("[project]\nname='w'\n")
    return str(r)


def call(env, payload):
    """One hook call through the dispatcher, on the DSH host."""
    out = io.StringIO()
    hook.main(["--host", "dsh"], io.StringIO(json.dumps(payload)), out, env)
    lines = out.getvalue().splitlines()
    return json.loads(lines[-1])


def store_for(env, repo):
    return Store(repo_state_dir(repo_identity(repo, env), "dsh", env))


def write_dsh_export(path, calls, model="qwen3.8-27b"):
    """One session's export in the exact shape the DSH bundle's adapter writes: `assistant`
    lines carrying `model` and a camelCase-derived `usage` breakdown (the bundle maps DSH's
    `inputTokens`/`outputTokens`/`cacheWriteTokens` to Claude's keys), and `user` lines with a
    `tool_result` per call. This is the shape the plugin's Claude-format readers are pointed at
    under `--host dsh`."""
    lines = []
    for i, (name, inp, res) in enumerate(calls):
        tid = f"t{i}"
        lines.append({"type": "assistant", "message": {
            "role": "assistant", "model": model,
            "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp},
                        {"type": "text", "text": f"{name} ran"}],
            "usage": {"input_tokens": 100, "output_tokens": 40,
                      "cache_creation_input_tokens": 10}}})
        lines.append({"type": "user", "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": res}]}})
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines))
    return path


def trap_export(path, repo, i):
    """One DSH session's export: the bundle's adapter's shape for the trap every session
    falls into (bare `pytest` missing, the working form next to it)."""
    calls = [("Edit", {"file_path": f"{repo}/pkg/mod{i}.py", "old_string": "a",
                       "new_string": "b"}, "ok"),
             ("Bash", {"command": "pytest -q tests/test_a.py"}, "bash: pytest: command not found"),
             ("Bash", {"command": "python -m pytest -q tests/test_a.py"}, "1 passed in 0.1s")]
    write_dsh_export(path, calls)
    return path


# -- registration and detection -------------------------------------------------------------

def test_the_dsh_host_is_registered_and_detected(env):
    assert isinstance(hosts.HOSTS["dsh"], type)
    assert isinstance(hosts.detect({}, env=env), hosts.Dsh)
    assert hosts.default_host(env=env) == "dsh"
    host = hosts.Dsh()
    assert host.name == "dsh"
    assert host.headless(env={"TE_DSH_MODE": "headless"})
    assert not host.headless(env={"TE_DSH_MODE": "tty"})
    assert not host.headless(env={})


def test_dsh_reads_its_export_with_the_claude_readers(tmp_path):
    path = trap_export(str(tmp_path / "s.jsonl"), "/repo", 1)
    host = hosts.Dsh()
    usage = host.session_usage(path)
    assert usage["requests"] > 0
    assert usage["input_tokens"] >= 0
    assert host.mechanism_trace(path, str(tmp_path)) == path
    assert os.path.getsize(path) > 0


# -- a repository life on the host ----------------------------------------------------------

def test_a_dsh_repository_folds_and_injects(env, repo, tmp_path):
    for i in range(6):
        base = {"session_id": f"s{i}", "transcript_path":
                trap_export(str(tmp_path / f"s{i}.jsonl"), repo, i), "cwd": repo}
        start = call(env, dict(base, hook_event_name="SessionStart", source="startup"))
        call(env, dict(base, hook_event_name="SessionEnd", reason="closed"))
        assert start == {} or "hookSpecificOutput" in start
    # The envfacts distiller is LLM-free: no `dsh` binary is needed for it to learn.
    st = store_for(env, repo)
    inj = st.read("injection.json")
    assert inj is not None and inj.get("rendered")
    assert int(inj.get("sessions", 0) or 0) == 6
    ledger = list(st.read_lines("ledger.jsonl"))
    assert any(r.get("kind") == "session" for r in ledger)


def test_the_dsh_host_never_reads_dshs_own_logs(tmp_path):
    """`keeps_history` stays off: the bundle reports every session it sees; there is no DSH
    folder layout to import from, and the plugin must not scan for one."""
    assert hosts.Dsh.keeps_history is False
    assert hosts.Dsh().history(env={}) == []


# -- the side call --------------------------------------------------------------------------

def _fake_dsh(tmp_path, stdout_lines, stderr=""):
    exe = tmp_path / "dsh"
    body = "#!/bin/sh\ncat > /dev/null   # the prompt arrives on stdin\n"
    body += "".join(f"printf '%s\\n' {json.dumps(line)}\n" for line in stdout_lines)
    if stderr:
        body += f"printf '%s' {json.dumps(stderr)} >&2\n"
    exe.write_text(body)
    os.chmod(exe, 0o755)
    return str(exe)


def test_dsh_json_reply_parses_the_headless_run():
    stdout = "\n".join([
        '{"type":"session","sessionId":"s1","cwd":"/tmp"}',
        '{"type":"status","phase":"turn_start","turn":1}',
        '{"type":"status","phase":"step_start","turn":1,"step":1}',
        '{"type":"model/selection","data":{"provider":"ninfer","model":"qwen3.8-27b"}}',
        '{"type":"thinking","text":"..."}',
        '{"type":"text","text":"pong"}',
        '{"type":"status","phase":"step_end","turn":1,"step":1,"usage":{"inputTokens":10,'
        '"outputTokens":5,"totalTokens":100,"cacheWriteTokens":7}}',
        '{"type":"status","phase":"step_end","turn":1,"step":2,"usage":{"inputTokens":4,'
        '"outputTokens":2,"totalTokens":30}}',
        '{"type":"status","phase":"turn_end","turn":1,"reason":{"kind":"completed"}}',
        '{"type":"final","text":"pong"}',
    ])
    text, usage, model = sidecall.dsh_json_reply(stdout)
    assert text == "pong"
    assert model == "qwen3.8-27b"
    assert usage == {"input_tokens": 14, "cache_read_input_tokens": 0,
                     "cache_creation_input_tokens": 7, "output_tokens": 7}


def test_a_dsh_side_call_runs_and_charges(tmp_path):
    exe = _fake_dsh(tmp_path, [
        '{"type":"status","phase":"step_end","turn":1,"step":1,"usage":{"inputTokens":100,'
        '"outputTokens":50,"totalTokens":300,"cacheWriteTokens":10}}',
        '{"type":"final","text":"a repair fact, stated once"}',
    ])
    text, eff = sidecall.dsh_call("distill this", cli=exe)
    assert text == "a repair fact, stated once"
    assert eff is not None and eff > 0          # the breakdown is priced and reported


@pytest.mark.real_sidecalls
def test_a_side_call_that_is_unsigned_in_pauses_the_host(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKENECONOMIZER_HOME", str(tmp_path))
    exe = _fake_dsh(tmp_path, [],
                    stderr='dsh: NO_ADAPTER: no adapter registered for provider "ninfer"')
    out = sidecall.dsh_call("anything", cli=exe)
    assert out[0] == ""
    pause = sidecall.paused("dsh", str(tmp_path))
    assert pause is not None
    assert "no adapter" in pause.get("reason", "")


def test_a_dsh_side_call_reports_its_unsigned_in_provider():
    """The signed-out detector recognizes DSH's own wording, so a route with no key pauses
    instead of re-failing every tick."""
    out = type("Out", (), {"stdout": "", "stderr":
                           'dsh: NO_ADAPTER: no adapter registered for provider "ninfer"',
                           "returncode": 1})()
    assert sidecall._SIGNED_OUT_RE.search(out.stderr)
