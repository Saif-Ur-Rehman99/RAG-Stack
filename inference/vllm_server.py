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
_READY_TIMEOUT = 300   # 5 min


class VLLMEmbeddingServer:
    """
    Manages a vLLM embedding server via Docker.

    vLLM uses PyTorch with PTX JIT-compilation, so it works on newer GPU
    architectures (e.g. Blackwell SM 12.0) even without a dedicated image.

    Exposes an OpenAI-compatible API:
      POST /v1/embeddings  — embed texts
      GET  /health         — liveness probe
      GET  /v1/models      — model info
    """

    def __init__(self) -> None:
        self._model_id = settings.TEXT_EMBEDDING_MODEL_ID
        self._port = settings.VLLM_PORT
        self._image = settings.VLLM_IMAGE
        self._container = settings.VLLM_CONTAINER_NAME
        self._hf_token = settings.HF_TOKEN or ""
        self._hf_cache = Path.home() / ".cache" / "huggingface"
        self._sudo_pw = ""
        self._proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._is_healthy():
            logger.info("vLLM server already running at port %s.", self._port)
            return

        self._prompt_sudo()
        self._assert_docker_running()
        self._stop_existing_container()
        self._pull_image_if_missing()
        self._launch()
        self._register_signal_handlers()
        self._wait_until_healthy()

    def stop(self) -> None:
        logger.info("Stopping vLLM container '%s' ...", self._container)
        self._sudo(["docker", "stop", self._container], capture_output=True)
        if self._proc:
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        logger.info("vLLM server stopped.")

    def stream_logs(self) -> None:
        """Block and forward container stdout to the terminal until Ctrl-C."""
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
    # Docker helpers
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

        logger.info("Docker daemon not running — starting it ...")
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
            "Check: sudo systemctl status docker"
        )

    def _stop_existing_container(self) -> None:
        self._sudo(["docker", "rm", "-f", self._container], capture_output=True)

    def _pull_image_if_missing(self) -> None:
        result = self._sudo(["docker", "image", "inspect", self._image], capture_output=True)
        if result.returncode == 0:
            logger.info("Image %s already present locally.", self._image)
            return
        logger.info("Pulling %s (this may take a few minutes) ...", self._image)
        self._sudo(["docker", "pull", self._image])

    def _launch(self) -> None:
        self._hf_cache.mkdir(parents=True, exist_ok=True)

        docker_cmd = [
            "sudo", "-S", "docker", "run",
            "--name", self._container,
            "--gpus", "all",
            "--dns", "8.8.8.8",
            "--dns", "8.8.4.4",
            "-p", f"{self._port}:8000",
            "-v", f"{self._hf_cache}:/root/.cache/huggingface",
            "-e", f"HF_TOKEN={self._hf_token}",
            "--ipc=host",         # needed for efficient shared memory between vLLM workers
            "--rm",
            self._image,
            self._model_id,                    # model as positional arg
            "--runner", "pooling",             # pooling mode (replaces --task embed)
            "--port", "8000",
            "--host", "0.0.0.0",
        ]

        logger.info(
            "Launching vLLM embedding server [model=%s, port=%s] ...",
            self._model_id, self._port,
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
            r = requests.get(f"http://localhost:{self._port}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _wait_until_healthy(self) -> None:
        logger.info("Waiting for vLLM server to become ready ...")
        deadline = time.time() + _READY_TIMEOUT
        last_log = time.time()

        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                out = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(f"vLLM server process exited unexpectedly.\n{out}")

            if self._is_healthy():
                elapsed = int(time.time() - (deadline - _READY_TIMEOUT))
                logger.info("vLLM server ready at http://localhost:%s/v1  (%ds)", self._port, elapsed)
                return

            if time.time() - last_log >= 20:
                elapsed = int(time.time() - (deadline - _READY_TIMEOUT))
                logger.info("Still starting ... (%ds elapsed)", elapsed)
                last_log = time.time()

            time.sleep(_POLL_INTERVAL)

        self.stop()
        raise TimeoutError(f"vLLM server did not become ready within {_READY_TIMEOUT}s.")

    def _register_signal_handlers(self) -> None:
        def _handler(*_):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
