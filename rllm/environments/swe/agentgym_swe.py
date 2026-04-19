"""AgentGym SWE Environment — uses agentgym LofiSandbox as the sandbox backend.

Drop-in replacement for SWEEnv that works without a local Docker daemon.
Register as ``agentgym_swe`` in env_agent_mappings.py and use with
``rllm.env.name=agentgym_swe`` in training configs.
"""

import asyncio
import concurrent.futures
import inspect
import json
import logging
import re

import numpy as np
from datasets import Dataset, load_dataset

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image resolution: mshrl DOCKER_CONFIG
# ---------------------------------------------------------------------------

DEFAULT_WORKING_DIR = "/testbed"
REWARD_TIMEOUT = 300


def _docker_tag_gen(input_string: str) -> str:
    """Sanitise a string into a valid Docker tag (ported from mshrl)."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_-]", "", input_string.lower()))


# Mirrors mshrl/examples/k2_agent/envs/swe/dataset.py — DOCKER_CONFIG.
# Each value: (registry, image_template, default_working_dir).
# image_template is a *format string* with ``{id}`` placeholder for instance_id.
DOCKER_CONFIG: dict[str, dict] = {
    "swebench-verified": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swe-verified:{id_lower}",
        "working_dir": "/testbed",
    },
    "gym": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/agent",
        "image": "swe-gym:{id_s}",
        "working_dir": "/testbed",
    },
    "r2e": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "r2e_gym:{id}",
        "working_dir": "/testbed",
    },
    "rebench": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swe-rebench:{id}",
        "working_dir": "/testbed",
    },
    "multilingual": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swebench-multiligual:{id_lower}",
        "working_dir": "/testbed",
    },
    "swe-factory": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swe_factory:{id}",
        "working_dir": "/testbed",
    },
    "swe-factory-v1": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swe_factory:{id}",  # may be patched with language below
        "working_dir": "/testbed",
    },
    "swe-factory-v1-add-dependency": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swe_factory_add_dependency:{id}",
        "working_dir": "/testbed",
    },
    "swe-bench-pro": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swebench_pro:{id}",
        "working_dir": "/app",
    },
    "extra": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swebench-extra-4k:{id_tag}",
        "working_dir": "/home/user/repo",
    },
    "swe-rebench-v2": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "swerebenchv2-tested:{id}",
        "working_dir": "/testbed",  # overridden per-entry below
    },
    "multi-swe": {
        "registry": "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
        "image": "multi_swe:{id_lower}",
        "working_dir": "/testbed",  # overridden per-entry below
    },
}

# Some image overrides (ported from mshrl).
_IMAGE_OVERRIDES: dict[tuple[str, str], str] = {
    ("swebench-verified", "astropy__astropy-8707"): (
        "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment/swe_verified_v2:astropy__astropy-8707"
    ),
    ("swebench-verified", "astropy__astropy-8872"): (
        "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment/swe_verified_v2:astropy__astropy-8872"
    ),
}

# Map dataset names that appear in data to DOCKER_CONFIG keys.
_DATASET_ALIASES: dict[str, str] = {
    "r2e-gym": "r2e",
    "swebench_verified": "swebench-verified",
    "swe_bench_verified": "swebench-verified",
}


def resolve_image_and_workdir(
    dataset_name: str, instance_id: str, data: dict
) -> tuple[str, str]:
    """Resolve container image + working directory from mshrl-style fields.

    Mirrors ``resolve_docker_image_and_workspace`` in mshrl.
    """
    # Normalise dataset name via alias table.
    key = _DATASET_ALIASES.get(dataset_name, dataset_name)
    if key not in DOCKER_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}' (resolved key '{key}'). "
            f"Known: {sorted(DOCKER_CONFIG)}"
        )

    cfg = DOCKER_CONFIG[key]
    registry = cfg["registry"]

    # Build image name with template substitutions.
    image_tpl = cfg["image"]
    image_name = image_tpl.format(
        id=instance_id,
        id_lower=instance_id.lower(),
        id_s=instance_id.replace("__", "_s_"),
        id_tag=_docker_tag_gen(instance_id),
    )

    # Special cases ported from mshrl.
    if key == "swe-factory-v1":
        lang = data.get("language", "")
        if lang:
            image_name = image_name.replace("swe_factory:", f"swe_factory_{lang}:")

    override = _IMAGE_OVERRIDES.get((key, instance_id))
    full_image = override if override else f"{registry}/{image_name}"

    # Working directory overrides.
    if key == "multi-swe":
        repo = data.get("repo", "")
        working_dir = f'/home/{repo.split("/")[-1]}' if repo else DEFAULT_WORKING_DIR
    elif key == "swe-rebench-v2":
        repo = data.get("repo", "")
        working_dir = f'/{repo.split("/")[-1]}' if repo else DEFAULT_WORKING_DIR
    else:
        working_dir = cfg.get("working_dir", DEFAULT_WORKING_DIR)

    return full_image, working_dir


def resolve_image_from_dockerhub(
    docker_image: str,
    registry: str = "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment",
) -> str:
    """Fallback: strip DockerHub user prefix and prepend internal registry."""
    image_name = docker_image.split("/", 1)[-1]
    return f"{registry}/{image_name}"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AgentGymSWEEnv(BaseEnv):
    """SWE environment backed by agentgym ``LofiSandbox``.

    Exposes the standard ``BaseEnv`` synchronous interface (``reset``, ``step``,
    ``close``) while internally driving the async agentgym API.
    """

    def __init__(
        self,
        entry: dict | None = None,
        dataset: Dataset | None = None,
        idx: int | None = None,
        step_timeout: int = 30,
        reward_timeout: int = REWARD_TIMEOUT,
        ttl: int = 14400,
        wait_ready_timeout: int = 7200,
        scaffold: str = "r2egym",
        image_registry: str | None = None,
    ):
        if entry is not None:
            self.entry = entry
            self.dataset = None
            self.idx = None
        else:
            if dataset is None:
                dataset = load_dataset("R2E-Gym/R2E-Gym-Lite", split="test")
            self.dataset = dataset
            if idx is None:
                idx = np.random.randint(0, len(self.dataset))
            assert 0 <= idx < len(self.dataset), "Selected index out of range"
            self.idx = idx
            self.entry = self.dataset[idx]

        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.ttl = ttl
        self.wait_ready_timeout = wait_ready_timeout
        self.scaffold = scaffold
        self.image_registry = image_registry
        self.total_steps = 0
        self.sandbox = None
        self._file_backups: dict[str, list[str]] = {}

        assert scaffold in ("r2egym", "sweagent"), (
            f"Invalid scaffold: {scaffold}, must be 'r2egym' or 'sweagent'"
        )

    # ------------------------------------------------------------------
    # Async ↔ sync bridge
    # ------------------------------------------------------------------

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily create a persistent event loop bound to this env instance.

        Using ``asyncio.run()`` every call would close the loop after each
        coroutine — but ``LofiSandbox`` caches an aiohttp session bound to the
        loop it was first started on, so subsequent calls would hit
        ``RuntimeError: Event loop is closed``. Keeping one loop per env
        solves this.
        """
        loop = getattr(self, "_loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            self._loop = loop
        return loop

    def _run_async(self, coro):
        """Run an async coroutine from synchronous code.

        Uses a persistent per-env loop so that long-lived async resources
        (aiohttp session, sandbox ctrl) keep working across calls.
        """
        loop = self._get_loop()
        if loop.is_running():
            # Caller is inside a running loop (e.g. Jupyter) — spawn a helper
            # thread with its own fresh loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[str, dict]:
        problem_statement = self._run_async(self._async_reset())
        return problem_statement, {}

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        return self._run_async(self._async_step(action))

    def compute_final_reward(self) -> float:
        return self._run_async(self._async_compute_final_reward())

    def close(self) -> None:
        if self.sandbox is not None:
            self._run_async(self._async_close())
        # Close the persistent loop if any
        loop = getattr(self, "_loop", None)
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    @staticmethod
    def from_dict(extra_info: dict | str) -> "AgentGymSWEEnv":
        """Create an instance from a dataset row dict."""
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(AgentGymSWEEnv.__init__)
        init_params: dict = {}
        for param_name in sig.parameters:
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = extra_info
        return AgentGymSWEEnv(**init_params)

    # ------------------------------------------------------------------
    # Image resolution
    # ------------------------------------------------------------------

    def _resolve_image_and_workdir(self) -> tuple[str, str]:
        """Determine container image and working directory for this entry.

        Priority:
        1. ``dataset`` + ``instance_id`` → mshrl DOCKER_CONFIG lookup.
        2. ``docker_image`` → strip DockerHub prefix, add internal registry.
        3. ``image`` field used as-is.
        """
        dataset_name = self.entry.get("dataset", "")
        instance_id = self.entry.get("instance_id", "")

        # Path 1: mshrl-style (dataset + instance_id).
        if dataset_name and instance_id:
            try:
                return resolve_image_and_workdir(dataset_name, instance_id, self.entry)
            except ValueError:
                logger.warning(
                    "DOCKER_CONFIG lookup failed for dataset=%s, falling back.",
                    dataset_name,
                )

        # Path 2: DockerHub docker_image field.
        docker_image = self.entry.get("docker_image", "")
        if docker_image:
            registry = (
                self.image_registry
                or "msh-m2-registry-vpc.ap-southeast-1.cr.aliyuncs.com/alignment"
            )
            image = resolve_image_from_dockerhub(docker_image, registry)
            working_dir = self.entry.get("workspace_dir", DEFAULT_WORKING_DIR)
            return image, working_dir

        # Path 3: pre-resolved image field.
        image = self.entry.get("image", "")
        if image:
            return image, self.entry.get("workspace_dir", DEFAULT_WORKING_DIR)

        raise ValueError(
            "Cannot resolve container image: entry has none of "
            "'dataset'+'instance_id', 'docker_image', or 'image'."
        )

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def _async_reset(self) -> str:
        from agentgym.sandbox.lofi import LofiSandbox

        # Tear down a previous sandbox.
        if self.sandbox is not None:
            try:
                await self.sandbox.stop()
            except Exception:
                pass
            self.sandbox = None

        image, working_dir = self._resolve_image_and_workdir()

        logger.info("Starting sandbox: image=%s  working_dir=%s", image, working_dir)
        self.sandbox = LofiSandbox(
            wait_ready_timeout=self.wait_ready_timeout,
            ttl=self.ttl,
            configs={
                "raw_image": image,
                "working_dir": working_dir,
                "cap_add": None,
            },
        )
        await self.sandbox.start()
        logger.info("Sandbox started successfully.")

        self._file_backups = {}
        self.total_steps = 0

        return self.entry.get("problem_statement", "")

    async def _async_step(self, action_str: str) -> tuple[str, float, bool, dict]:
        from rllm.environments.swe.agentgym_tools import execute_action, parse_xml_action

        function_name, params = parse_xml_action(action_str)
        if not function_name:
            return (
                "Could not parse action. Please use the correct XML format:\n"
                "<function=name><parameter=key>value</parameter></function>",
                0,
                False,
                {},
            )

        observation, done = await execute_action(
            self.sandbox,
            function_name,
            params,
            self._file_backups,
            self.step_timeout,
        )

        self.total_steps += 1
        return observation, 0, done, {}

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    async def _async_compute_final_reward(self) -> float:
        """Compute reward by running tests inside the sandbox.

        Strategy (by priority):
        1. ``dataset`` field → delegate to mshrl ``final_judge`` pattern.
        2. ``expected_output_json`` → R2E-Gym Subset style.
        3. ``instance_id`` with swebench fields → swebench harness.
        4. Return 0.
        """
        dataset_name = self.entry.get("dataset", "")
        instance_id = self.entry.get("instance_id", "")

        # Path 1: mshrl-style judge (dataset + instance_id).
        if dataset_name and instance_id:
            try:
                return await self._reward_mshrl_judge()
            except Exception as e:
                logger.error("mshrl judge failed: %s", e, exc_info=True)
                return 0.0

        # Path 2: R2E-Gym Subset (expected_output_json).
        expected_output = self.entry.get("expected_output_json", None)
        if expected_output:
            return await self._reward_from_expected_output(expected_output)

        # Path 3: swebench harness (needs instance_id + FAIL_TO_PASS etc.).
        if instance_id:
            return await self._reward_from_swebench()

        logger.warning("No evaluation method available for this entry.")
        return 0.0

    # -- mshrl judge (mirrors judge.py) ----------------------------------

    async def _reward_mshrl_judge(self) -> float:
        """Run evaluation matching the mshrl ``final_judge`` dispatcher."""
        dataset_name = self.entry["dataset"]
        key = _DATASET_ALIASES.get(dataset_name, dataset_name)

        if key in ("swebench-verified", "gym"):
            report = await self._judge_swebench_verified()
        elif key == "r2e":
            report = await self._judge_r2e_gym()
        elif key == "rebench":
            report = await self._judge_swebench_harness(
                "swebench.harness.grading_swe_rebench",
                "swebench.harness.test_spec_swe_rebench",
            )
        elif key == "multilingual":
            report = await self._judge_swebench_harness(
                "swebench.harness_multilingual.grading",
                "swebench.harness_multilingual.test_spec",
            )
        elif key in ("swe-factory", "swe-factory-v1", "swe-factory-v1-add-dependency"):
            report = await self._judge_swe_factory()
        elif key == "swe-bench-pro":
            report = await self._judge_swebench_pro()
        elif key == "swe-rebench-v2":
            report = await self._judge_swe_rebench_v2()
        elif key == "multi-swe":
            report = await self._judge_swebench_harness(
                "swebench.harness_multilingual.grading_multi_swe",
                "swebench.harness_multilingual.test_spec_multi_swe",
            )
        else:
            logger.warning("No judge for dataset=%s, returning 0.", dataset_name)
            return 0.0

        resolved = report.get("resolved", False)
        logger.info(
            "Judge result: dataset=%s  instance_id=%s  resolved=%s",
            dataset_name, self.entry.get("instance_id"), resolved,
        )
        return 1.0 if resolved else 0.0

    async def _judge_swebench_verified(self) -> dict:
        from swebench.harness.constants import SWEbenchInstance
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec import make_test_spec

        test_spec = make_test_spec(SWEbenchInstance(**self.entry))
        return await self._run_eval_script_and_grade(
            test_spec.eval_script,
            lambda log: get_eval_report(
                test_spec=test_spec,
                prediction={"instance_id": self.entry["instance_id"], "model_patch": "-1"},
                log_raw_content=log,
                include_tests_status=True,
            )[self.entry["instance_id"]],
        )

    async def _judge_r2e_gym(self) -> dict:
        from swebench.harness.constants import SWEbenchInstance
        from swebench.harness.grading_r2e_gym import get_eval_report
        from swebench.harness.test_spec import make_test_spec

        test_spec = make_test_spec(SWEbenchInstance(**self.entry))
        return await self._run_eval_script_and_grade(
            test_spec.eval_script,
            lambda log: get_eval_report(
                test_spec=test_spec,
                instance_id=self.entry["instance_id"],
                log_raw_content=log,
                include_tests_status=True,
            )[self.entry["instance_id"]],
        )

    async def _judge_swebench_harness(
        self, grading_module: str, test_spec_module: str
    ) -> dict:
        """Generic swebench harness judge."""
        import importlib
        grading = importlib.import_module(grading_module)
        ts_mod = importlib.import_module(test_spec_module)

        from swebench.harness.constants import SWEbenchInstance
        test_spec = ts_mod.make_test_spec(SWEbenchInstance(**self.entry))

        return await self._run_eval_script_and_grade(
            test_spec.eval_script,
            lambda log: grading.get_eval_report(
                test_spec=test_spec,
                instance_id=self.entry["instance_id"],
                log_raw_content=log,
                include_tests_status=True,
            )[self.entry["instance_id"]],
        )

    async def _judge_swe_factory(self) -> dict:
        eval_script = self.entry.get("eval_sh", "")
        # Offline patches (from mshrl judge.py).
        for old, new in [
            ("./gradlew test", "./gradlew test --offline"),
            ("./gradlew :", "./gradlew --offline :"),
            ("gradle test", "gradle test --offline"),
            ("mvn test", "mvn -o test"),
            ("./mvnw test", "./mvnw -o test"),
        ]:
            eval_script = eval_script.replace(old, new)

        await self.sandbox.ctrl.write_text("/eval.sh", eval_script)
        val = await self.sandbox.ctrl.run(
            ["/bin/bash", "/eval.sh"],
            capture_output=True, timeout=self.reward_timeout,
            text=True, raise_on_timeout=False,
        )
        log_str = (val.stderr or "") + (val.stdout or "")
        match = re.search(r"echo OMNIGRIL_EXIT_CODE=(\d)", log_str)
        resolved = bool(match and match.group(1) == "0")
        return {"resolved": resolved, "raw_log_content": log_str}

    async def _judge_swebench_pro(self) -> dict:
        from swebench.harness_multilingual.grading_swebench_pro import get_eval_report
        from swebench.harness_multilingual.test_spec_swebench_pro import make_test_spec

        await self.sandbox.ctrl.mkdir("/workspace", parents=True)
        await self.sandbox.ctrl.write_text("/workspace/run_script.sh", self.entry["run_script"])
        await self.sandbox.ctrl.write_text("/workspace/parser.py", self.entry["parser"])

        test_spec = make_test_spec(self.entry)
        await self.sandbox.ctrl.write_text("/workspace/entryscript.sh", test_spec.eval_script)
        val = await self.sandbox.ctrl.run(
            ["/bin/bash", "/workspace/entryscript.sh"],
            capture_output=True, timeout=self.reward_timeout,
            text=True, raise_on_timeout=False,
        )
        log_str = (val.stderr or "") + (val.stdout or "")
        try:
            output_json_text = await self.sandbox.ctrl.read_text("/workspace/output.json")
            output_json = json.loads(output_json_text)
            report = get_eval_report(test_spec, output_json["tests"])
            return {"resolved": report["resolved"], "raw_log_content": log_str}
        except Exception:
            return {"resolved": False, "raw_log_content": log_str}

    async def _judge_swe_rebench_v2(self) -> dict:
        from swebench.harness_swerebenchv2.grading import get_eval_report
        from swebench.harness_swerebenchv2.test_spec import make_test_spec

        test_spec = make_test_spec(self.entry)
        p = await self.sandbox.ctrl.run(
            ["/bin/bash", "-c", test_spec.eval_script],
            capture_output=True, timeout=self.reward_timeout,
            text=True, raise_on_timeout=False,
        )
        output = (p.stdout or "") + (p.stderr or "")
        return get_eval_report(test_spec, output)

    async def _run_eval_script_and_grade(self, eval_script: str, grade_fn) -> dict:
        """Write eval script to sandbox, run it, and grade the output."""
        await self.sandbox.ctrl.write_text("/eval.sh", eval_script)
        val = await self.sandbox.ctrl.run(
            ["/bin/bash", "/eval.sh"],
            capture_output=True, timeout=self.reward_timeout,
            text=True, raise_on_timeout=False,
        )
        log_str = (val.stderr or "") + (val.stdout or "")
        return grade_fn(log_str)

    # -- R2E-Gym Subset (expected_output_json) ----------------------------

    async def _reward_from_expected_output(self, expected_output) -> float:
        """Evaluate using ``expected_output_json`` (R2E-Gym Subset)."""
        if isinstance(expected_output, str):
            try:
                expected_output = json.loads(expected_output)
            except json.JSONDecodeError:
                logger.error("Failed to parse expected_output_json.")
                return 0.0

        if not expected_output:
            return 0.0

        test_names = list(expected_output.keys())
        if not test_names:
            return 0.0

        class_names = {name.split(".")[0] for name in test_names if "." in name}
        if class_names:
            k_expr = " or ".join(sorted(class_names))
            k_flag = f'-k "{k_expr}"'
        else:
            k_flag = ""

        cmd = f"cd /testbed && python -m pytest {k_flag} -v --tb=no 2>&1 || true"

        try:
            result = await self.sandbox.ctrl.run(
                ["/bin/bash", "-c", cmd],
                capture_output=True, timeout=self.reward_timeout,
                text=True, raise_on_timeout=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return 0.0

        actual_results: dict[str, bool] = {}
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            for status_word in ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS"):
                if line.endswith(status_word):
                    node_id = line[: -len(status_word)].strip()
                    passed = status_word in ("PASSED", "XFAIL")
                    for test_name in test_names:
                        colonized = test_name.replace(".", "::")
                        if colonized in node_id or test_name in node_id:
                            actual_results[test_name] = passed
                    break

        total = len(expected_output)
        matched = 0
        for test_name, expected_status in expected_output.items():
            expected_pass = expected_status.upper() == "PASSED"
            actual_pass = actual_results.get(test_name)
            if actual_pass is None:
                logger.debug(f"Test '{test_name}' not found in pytest output.")
                continue
            if actual_pass == expected_pass:
                matched += 1

        logger.info("Reward: %d/%d tests matched expected results.", matched, total)
        return 1.0 if matched == total else 0.0

    # -- swebench harness fallback ----------------------------------------

    async def _reward_from_swebench(self) -> float:
        """Fallback for entries that have instance_id + swebench fields."""
        try:
            report = await self._judge_swebench_verified()
            return 1.0 if report["resolved"] else 0.0
        except Exception as e:
            logger.error(f"swebench evaluation failed: {e}", exc_info=True)
            return 0.0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _async_close(self):
        if self.sandbox is not None:
            try:
                await self.sandbox.stop()
            except Exception as e:
                logger.warning(f"Error stopping sandbox: {e}")
            self.sandbox = None
