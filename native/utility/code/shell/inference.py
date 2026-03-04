from inferencesh import BaseApp, BaseAppSetup
from pydantic import BaseModel, Field
from typing import List, Optional
import subprocess
import os


class AppSetup(BaseAppSetup):
    """Shell app configuration."""
    working_dir: str = Field(default=".", description="Default working directory")
    timeout: int = Field(default=30, description="Default timeout in seconds")


class RunInput(BaseModel):
    """Command execution input."""
    command: str = Field(description="Command to execute (passed to shell)")
    cwd: Optional[str] = Field(None, description="Working directory (overrides default)")
    timeout: Optional[int] = Field(None, description="Timeout in seconds (overrides default)")
    env: Optional[dict[str, str]] = Field(None, description="Additional environment variables")


class RunOutput(BaseModel):
    """Command execution result."""
    stdout: str = Field(description="Standard output")
    stderr: str = Field(description="Standard error")
    exit_code: int = Field(description="Exit code")


class App(BaseApp):

    async def setup(self, config: AppSetup):
        self.working_dir = config.working_dir
        self.timeout = config.timeout

    async def run(self, input_data: RunInput) -> RunOutput:
        """Execute a shell command."""
        cwd = input_data.cwd or self.working_dir
        timeout = input_data.timeout or self.timeout

        # Merge environment
        env = os.environ.copy()
        if input_data.env:
            env.update(input_data.env)

        try:
            result = subprocess.run(
                input_data.command,
                shell=True,
                cwd=cwd,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            return RunOutput(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired as e:
            return RunOutput(
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                exit_code=-1,
            )
