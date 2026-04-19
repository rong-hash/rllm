"""XML action parsing and sandbox execution tools for AgentGym SWE environment.

Translates SWEAgent XML actions (e.g. <function=execute_bash><parameter=cmd>ls</parameter></function>)
into operations on an agentgym LofiSandbox.
"""

import logging
import re

logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 50000


def parse_xml_action(action_str: str) -> tuple[str, dict]:
    """Parse XML action format used by SWEAgent.

    Format: <function=name><parameter=key>value</parameter></function>

    Returns:
        (function_name, parameters_dict). Returns ("", {}) if parsing fails.
    """
    if not action_str or not action_str.strip():
        return "", {}

    func_match = re.search(r"<function=(\w+)>", action_str)
    if not func_match:
        return "", {}

    function_name = func_match.group(1)

    params = {}
    param_pattern = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
    for match in param_pattern.finditer(action_str):
        params[match.group(1)] = match.group(2)

    return function_name, params


def _clip_output(output: str) -> str:
    if len(output) > MAX_OUTPUT_LENGTH:
        return output[:MAX_OUTPUT_LENGTH] + "\n<response clipped>"
    return output


async def execute_action(
    sandbox,
    function_name: str,
    params: dict,
    file_backups: dict,
    step_timeout: int = 30,
) -> tuple[str, bool]:
    """Execute a parsed action in the sandbox.

    Args:
        sandbox: LofiSandbox instance.
        function_name: Parsed function name.
        params: Parsed parameters dict.
        file_backups: Mutable dict tracking file contents for undo_edit.
        step_timeout: Timeout in seconds for each command.

    Returns:
        (observation_str, done)
    """
    try:
        if function_name == "execute_bash":
            return await _execute_bash(sandbox, params, step_timeout)
        elif function_name in ("file_editor", "str_replace_editor"):
            return await _file_editor(sandbox, params, file_backups, step_timeout)
        elif function_name == "search":
            return await _search(sandbox, params, step_timeout)
        elif function_name in ("finish", "submit"):
            return _finish(params)
        else:
            return f"Unknown function: {function_name}", False
    except Exception as e:
        logger.error(f"Error executing {function_name}: {e}", exc_info=True)
        return f"Error executing {function_name}: {e}", False


# ---------------------------------------------------------------------------
# execute_bash
# ---------------------------------------------------------------------------

async def _execute_bash(sandbox, params: dict, timeout: int) -> tuple[str, bool]:
    cmd = params.get("cmd", params.get("command", ""))
    if not cmd:
        return "Error: No command provided for execute_bash.", False

    result = await sandbox.ctrl.run(
        ["/bin/bash", "-c", cmd],
        capture_output=True,
        timeout=timeout,
        text=True,
        raise_on_timeout=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if not output:
        output = f"Command executed successfully (exit code: {result.returncode})"
    return _clip_output(output), False


# ---------------------------------------------------------------------------
# file_editor / str_replace_editor
# ---------------------------------------------------------------------------

async def _file_editor(
    sandbox, params: dict, file_backups: dict, timeout: int
) -> tuple[str, bool]:
    command = params.get("command", "")
    path = params.get("path", "")

    if not command:
        return "Error: No command specified for file_editor.", False
    if not path:
        return "Error: No path specified for file_editor.", False

    if command == "view":
        return await _file_view(sandbox, path, params, timeout)
    elif command == "create":
        return await _file_create(sandbox, path, params, file_backups, timeout)
    elif command == "str_replace":
        return await _file_str_replace(sandbox, path, params, file_backups, timeout)
    elif command == "insert":
        return await _file_insert(sandbox, path, params, file_backups, timeout)
    elif command == "undo_edit":
        return await _file_undo_edit(sandbox, path, file_backups)
    else:
        return f"Error: Unknown file_editor command: {command}", False


async def _file_view(
    sandbox, path: str, params: dict, timeout: int
) -> tuple[str, bool]:
    """View file or directory contents."""
    check = await sandbox.ctrl.run(
        ["/bin/bash", "-c", f'test -d "{path}" && echo DIR || echo FILE'],
        capture_output=True,
        timeout=timeout,
        text=True,
        raise_on_timeout=False,
    )
    is_dir = check.stdout.strip() == "DIR"

    if is_dir:
        result = await sandbox.ctrl.run(
            [
                "/bin/bash",
                "-c",
                f'find "{path}" -maxdepth 2 -not -path "*/\\.*" | head -200',
            ],
            capture_output=True,
            timeout=timeout,
            text=True,
            raise_on_timeout=False,
        )
        return result.stdout or "Empty directory", False

    # File view
    view_range = params.get("view_range", "")
    if view_range:
        range_match = re.match(r"\[?\s*(\d+)\s*,\s*(-?\d+)\s*]?", view_range)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end == -1:
                cmd = f'cat -n "{path}" | tail -n +{start}'
            else:
                count = end - start + 1
                cmd = f'cat -n "{path}" | tail -n +{start} | head -n {count}'
        else:
            cmd = f'cat -n "{path}"'
    else:
        cmd = f'cat -n "{path}"'

    result = await sandbox.ctrl.run(
        ["/bin/bash", "-c", cmd],
        capture_output=True,
        timeout=timeout,
        text=True,
        raise_on_timeout=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return _clip_output(output) or f"Error: Could not read file {path}", False


async def _file_create(
    sandbox, path: str, params: dict, file_backups: dict, timeout: int
) -> tuple[str, bool]:
    """Create a new file (fails if file already exists)."""
    file_text = params.get("file_text", "")

    check = await sandbox.ctrl.run(
        [
            "/bin/bash",
            "-c",
            f'if [ -d "{path}" ]; then echo DIR; elif [ -e "{path}" ]; then echo EXISTS; else echo NEW; fi',
        ],
        capture_output=True,
        timeout=timeout,
        text=True,
        raise_on_timeout=False,
    )
    status = check.stdout.strip()
    if status == "DIR":
        return f"Error: Path {path} is a directory, cannot create a file with that name.", False
    if status == "EXISTS":
        return f"Error: File already exists at {path}. Use str_replace to edit.", False

    # Ensure parent directory exists
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        await sandbox.ctrl.run(
            ["/bin/bash", "-c", f'mkdir -p "{parent}"'],
            capture_output=True,
            timeout=timeout,
            text=True,
            raise_on_timeout=False,
        )

    await sandbox.ctrl.write_text(path, file_text)
    file_backups.setdefault(path, [])
    return f"File created successfully at: {path}", False


async def _file_str_replace(
    sandbox, path: str, params: dict, file_backups: dict, timeout: int
) -> tuple[str, bool]:
    """Replace a unique occurrence of old_str with new_str in a file."""
    old_str = params.get("old_str", "")
    new_str = params.get("new_str", "")

    if not old_str:
        return "Error: old_str is required for str_replace.", False

    try:
        content = await sandbox.ctrl.read_text(path)
    except Exception as e:
        return f"Error reading file {path}: {e}", False

    count = content.count(old_str)
    if count == 0:
        return (
            f"Error: old_str not found in {path}. "
            "Ensure it matches the file content exactly, including whitespace."
        ), False
    if count > 1:
        return (
            f"Error: old_str found {count} times in {path}. "
            "Include more surrounding context to make it unique."
        ), False

    file_backups.setdefault(path, []).append(content)
    new_content = content.replace(old_str, new_str, 1)
    await sandbox.ctrl.write_text(path, new_content)
    return f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n", False


async def _file_insert(
    sandbox, path: str, params: dict, file_backups: dict, timeout: int
) -> tuple[str, bool]:
    """Insert new_str after insert_line in a file."""
    new_str = params.get("new_str", "")
    insert_line_str = params.get("insert_line", "")

    if not new_str:
        return "Error: new_str is required for insert.", False
    if not insert_line_str:
        return "Error: insert_line is required for insert.", False

    try:
        line_num = int(insert_line_str)
    except ValueError:
        return f"Error: insert_line must be an integer, got '{insert_line_str}'.", False

    try:
        content = await sandbox.ctrl.read_text(path)
    except Exception as e:
        return f"Error reading file {path}: {e}", False

    file_backups.setdefault(path, []).append(content)

    lines = content.split("\n")
    line_num = max(0, min(line_num, len(lines)))
    new_lines = new_str.split("\n")
    lines = lines[:line_num] + new_lines + lines[line_num:]

    await sandbox.ctrl.write_text(path, "\n".join(lines))
    return f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n", False


async def _file_undo_edit(
    sandbox, path: str, file_backups: dict
) -> tuple[str, bool]:
    """Undo the last edit to a file."""
    if path not in file_backups or not file_backups[path]:
        return f"Error: No edit history found for {path}.", False

    previous_content = file_backups[path].pop()
    await sandbox.ctrl.write_text(path, previous_content)
    return f"Last edit to {path} undone successfully.", False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

async def _search(sandbox, params: dict, timeout: int) -> tuple[str, bool]:
    """Search for a term in a file or directory."""
    search_term = params.get("search_term", "")
    path = params.get("path", ".")

    if not search_term:
        return "Error: search_term is required for search.", False

    # Escape double quotes in search term for shell safety
    escaped = search_term.replace("\\", "\\\\").replace('"', '\\"')
    result = await sandbox.ctrl.run(
        ["/bin/bash", "-c", f'grep -rn "{escaped}" "{path}" 2>/dev/null | head -200'],
        capture_output=True,
        timeout=timeout,
        text=True,
        raise_on_timeout=False,
    )
    output = result.stdout or ""
    if not output:
        return f"No matches found for '{search_term}' in {path}", False

    lines = output.strip().split("\n")
    if len(lines) >= 200:
        output += "\n<response clipped> Too many results. Narrow your search."
    return output, False


# ---------------------------------------------------------------------------
# finish / submit
# ---------------------------------------------------------------------------

def _finish(params: dict) -> tuple[str, bool]:
    """Handle finish/submit action."""
    result = params.get("result", params.get("command", ""))
    return result or "Submitted.", True
