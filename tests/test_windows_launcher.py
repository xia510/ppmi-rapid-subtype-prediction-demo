import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.request import urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
START_SCRIPT = PROJECT_DIR / "scripts" / "start_demo.ps1"
STOP_SCRIPT = PROJECT_DIR / "scripts" / "stop_demo.ps1"


@unittest.skipUnless(os.name == "nt", "Windows PowerShell launcher test")
class WindowsLauncherTests(unittest.TestCase):
    def test_api_key_prompt_uses_visible_text_input(self):
        script_text = START_SCRIPT.read_text(encoding="utf-8-sig")

        self.assertIn("Read-ApiKeyText", script_text)
        self.assertNotIn("-AsSecureString", script_text)

    def run_validation(self, env_text: str):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(env_text, encoding="utf-8")
            return subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(START_SCRIPT),
                    "-ValidateOnly",
                    "-NonInteractive",
                    "-EnvFile",
                    str(env_path),
                ],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )

    def test_validation_accepts_local_env_without_printing_secret(self):
        secret = "sk-test-secret-must-not-be-printed"
        result = self.run_validation(
            "DEEPSEEK_API_KEY={secret}\nPYTHON_EXE={python}\n".format(
                secret=secret,
                python=sys.executable,
            )
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Configuration valid", result.stdout)
        self.assertNotIn(secret, result.stdout + result.stderr)

    def test_validation_rejects_blank_key_in_noninteractive_mode(self):
        result = self.run_validation(
            "DEEPSEEK_API_KEY=\nPYTHON_EXE={python}\n".format(python=sys.executable)
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DEEPSEEK_API_KEY", result.stdout + result.stderr)

    def test_validation_rejects_nonempty_malformed_key(self):
        result = self.run_validation(
            "DEEPSEEK_API_KEY=x\nPYTHON_EXE={python}\n".format(python=sys.executable)
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DEEPSEEK_API_KEY", result.stdout + result.stderr)

    def test_env_file_is_parsed_as_data_not_executed_as_powershell(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist.txt"
            result = self.run_validation(
                "DEEPSEEK_API_KEY=sk-test-safe-123456789012345\n"
                "PYTHON_EXE={python}\n"
                "UNTRUSTED=$(Set-Content -LiteralPath '{marker}' -Value hacked)\n".format(
                    python=sys.executable,
                    marker=marker,
                )
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(marker.exists())

    def test_launcher_starts_and_stops_both_local_services(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "DEEPSEEK_API_KEY=sk-test-launcher-123456789012345\n"
                "HF_HUB_OFFLINE=1\n"
                "PYTHON_EXE={python}\n".format(python=sys.executable),
                encoding="utf-8",
            )
            start_command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(START_SCRIPT),
                "-EnvFile",
                str(env_path),
                "-NonInteractive",
                "-NoBrowser",
            ]
            stop_command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(STOP_SCRIPT),
                "-Quiet",
            ]

            try:
                started = subprocess.run(
                    start_command,
                    cwd=PROJECT_DIR,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                with urlopen("http://127.0.0.1:8501", timeout=5) as response:
                    self.assertEqual(response.status, 200)
            finally:
                stopped = subprocess.run(
                    stop_command,
                    cwd=PROJECT_DIR,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                )
            self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)


if __name__ == "__main__":
    unittest.main()
