import getpass
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from config import settings

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5
_READY_TIMEOUT = 1800   # 30 min — TRT engine compilation can take this long on first run


class TRTLLMServer:
    """
    Manages the TensorRT-LLM inference server lifecycle via Docker.

    Backends (set TRTLLM_BACKEND in .env):
      'pytorch'  — starts in seconds; no compilation; good for smoke testing
      'tensorrt' — JIT-compiles a TRT engine on first run (5-20 min);
                   subsequent runs reuse the cached engine and start in seconds
    """

    def __init__(self) -> None:
        self._model = settings.TRTLLM_MODEL
        self._port = settings.TRTLLM_HOST_PORT
        self._image = settings.TRTLLM_IMAGE
        self._container = settings.TRTLLM_CONTAINER_NAME
        self._backend = settings.TRTLLM_BACKEND
        self._max_batch_size = settings.TRTLLM_MAX_BATCH_SIZE
        self._max_num_tokens = settings.TRTLLM_MAX_NUM_TOKENS
        self._max_seq_len = settings.TRTLLM_MAX_SEQ_LEN
        self._hf_token = settings.HUGGINGFACE_ACCESS_TOKEN or ""
        self._hf_cache = Path.home() / ".cache" / "huggingface"
        self._engine_cache = Path.home() / ".cache" / "trtllm_engines"
        self._sudo_pw = ""
        self._proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._is_healthy():
            logger.info("TensorRT-LLM server already running at port %s.", self._port)
            return

        self._prompt_sudo()
        self._assert_docker_running()
        self._assert_gpu_accessible()
        self._stop_existing_container()
        self._pull_image_if_missing()
        self._launch()
        self._register_signal_handlers()
        self._wait_until_healthy()

    def stop(self) -> None:
        logger.info("Stopping container '%s' ...", self._container)
        self._sudo(["docker", "stop", self._container], capture_output=True)
        if self._proc:
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        logger.info("Server stopped.")

    def stream_logs(self) -> None:
        """Block and forward server stdout to the terminal until Ctrl-C."""
        if not self._proc or not self._proc.stdout:
            logger.warning("No server process to stream logs from.")
            return
        try:
            for line in self._proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        except KeyboardInterrupt:
            self.stop()

    # ------------------------------------------------------------------
    # Docker / system helpers
    # ------------------------------------------------------------------

    def _prompt_sudo(self) -> None:
        self._sudo_pw = getpass.getpass("sudo password: ")

    def _sudo(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sudo", "-S"] + cmd,
            input=self._sudo_pw + "\n",
            text=True,
            **kwargs,
        )

    def _assert_docker_running(self) -> None:
        if self._sudo(["docker", "info"], capture_output=True).returncode == 0:
            logger.info("Docker daemon is running.")
            return

        logger.info("Docker daemon not running — starting it (sudo required) ...")
        subprocess.run(["sudo", "systemctl", "start", "docker"])

        for attempt in range(30):
            time.sleep(2)
            if self._sudo(["docker", "info"], capture_output=True).returncode == 0:
                logger.info("Docker daemon ready (%ds).", (attempt + 1) * 2)
                return
            if attempt % 5 == 4:
                logger.info("Still waiting for Docker ... (%ds elapsed)", (attempt + 1) * 2)

        raise RuntimeError(
            "Docker daemon did not become ready within 60s.\n"
            "Check status: sudo systemctl status docker"
        )

    def _assert_gpu_accessible(self) -> None:
        logger.info("Checking GPU access inside Docker ...")
        result = self._sudo(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "GPU not accessible inside Docker.\n"
                "Install NVIDIA Container Toolkit:\n"
                "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
            )
        logger.info("GPU is accessible inside Docker.")

    def _stop_existing_container(self) -> None:
        self._sudo(["docker", "rm", "-f", self._container], capture_output=True)

    def _pull_image_if_missing(self) -> None:
        result = self._sudo(["docker", "image", "inspect", self._image], capture_output=True)
        if result.returncode == 0:
            logger.info("Image %s already present locally.", self._image)
            return
        logger.info("Pulling %s — this is several GB, please wait ...", self._image)
        self._sudo(["docker", "pull", self._image])

    def _launch(self) -> None:
        self._hf_cache.mkdir(parents=True, exist_ok=True)
        self._engine_cache.mkdir(parents=True, exist_ok=True)

        inner_cmd = (
            f"trtllm-serve {self._model} "
            f"--host 0.0.0.0 --port 8000 "
            f"--max_batch_size {self._max_batch_size} "
            f"--max_num_tokens {self._max_num_tokens} "
            f"--max_seq_len {self._max_seq_len} "
            f"--backend {self._backend}"
        )

        docker_cmd = [
            "sudo", "-S", "docker", "run",
            "--name", self._container,
            "--gpus", "all",
            "--ipc=host",                       # efficient multi-process tensor sharing
            "--ulimit", "memlock=-1",            # TRT pins GPU memory; remove default cap
            "--ulimit", "stack=67108864",
            "-p", f"{self._port}:8000",
            "-v", f"{self._hf_cache}:/root/.cache/huggingface",
            "-v", f"{self._engine_cache}:/root/.cache/trtllm",
            "-e", f"HF_TOKEN={self._hf_token}",
            "--rm",
            self._image,
            "bash", "-c", inner_cmd,
        ]

        logger.info(
            "Launching TensorRT-LLM [model=%s, backend=%s, port=%s] ...",
            self._model, self._backend, self._port,
        )
        if self._backend == "tensorrt":
            logger.info(
                "First run compiles the TRT engine (5–20 min). "
                "Subsequent runs reuse the cached engine and start in seconds."
            )

        self._proc = subprocess.Popen(
            docker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(self._sudo_pw + "\n")
        self._proc.stdin.flush()

    # ------------------------------------------------------------------
    # Health & lifecycle
    # ------------------------------------------------------------------

    def _is_healthy(self) -> bool:
        try:
            r = requests.get(
                f"http://localhost:{self._port}/v1/models", timeout=3
            )
            return r.status_code == 200
        except Exception:
            return False

    def _wait_until_healthy(self) -> None:
        logger.info(
            "Waiting for server to become ready (up to %d min) ...",
            _READY_TIMEOUT // 60,
        )
        deadline = time.time() + _READY_TIMEOUT
        last_log = time.time()

        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                out = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(f"Server process exited unexpectedly.\n{out}")

            if self._is_healthy():
                elapsed = int(time.time() - (deadline - _READY_TIMEOUT))
                logger.info(
                    "Server is ready at http://localhost:%s/v1  (%ds)", self._port, elapsed
                )
                return

            if time.time() - last_log >= 30:
                elapsed = int(time.time() - (deadline - _READY_TIMEOUT))
                logger.info("Still starting ... (%ds elapsed)", elapsed)
                last_log = time.time()

            time.sleep(_POLL_INTERVAL)

        self.stop()
        raise TimeoutError(
            f"Server did not become ready within {_READY_TIMEOUT // 60} minutes."
        )

    def _register_signal_handlers(self) -> None:
        def _handler(*_):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
