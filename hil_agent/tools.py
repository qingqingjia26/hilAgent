import os
import subprocess
import uuid
from datetime import datetime
from typing import List, Union

from langchain_core.tools import tool
from pydantic import BaseModel
from pydantic.fields import Field


@tool
def check_weather(location: str, at_time: str) -> str:
    """Return the weather forecast for the specified location."""
    print("------------------------------tool called-------------------------")
    print(
        f"check_weather called with location: {location} and at_time: {at_time}"
    )
    print(f"\n")
    return f"It's 25°C in {location} at {at_time}"


class ReqInput(BaseModel):
    """Input for the get_requirement tool."""

    req_name: str = Field(
        description="The name of the customized requirement to get",
    )


class Write2FileInput(BaseModel):
    """Input for the write2file tool."""

    filename: str = Field(
        description="The name of the file to write to",
    )
    content: str = Field(
        description="The content to write to the file",
    )


@tool(args_schema=Write2FileInput)
def write2file(filename: str, content: str) -> str:
    """Write content to a file."""
    print("------------------------------tool called-------------------------")
    print(f"write2file called with filename: {filename} and content: {content}")
    print(f"\n")
    try:
        req_dir = "./tmp"
        file_dir = f"{req_dir}/results"
        os.makedirs(file_dir, exist_ok=True)
        output_file = os.path.join(file_dir, filename)
        if os.path.exists(output_file):
            base, ext = os.path.splitext(filename)
            filename = f"{filename}_{uuid.uuid4()}"
            uuid_str = str(uuid.uuid4())
            output_file = os.path.join(file_dir, f"{base}_{uuid_str}.{ext}")
        with open(output_file, "w") as f:
            f.write(content)
    except Exception as e:
        return str(e)
    return f"Successfully written to {filename}"


g_time_limit = 10  # seconds


def _run_cmd_on_host(cmd: Union[str, List[str]]) -> str:
    input_cmd = cmd
    if isinstance(cmd, list):
        cmd = [item.strip() for item in cmd]
        input_cmd = " ; ".join(cmd)

    print("input_cmd: ", input_cmd)
    try:
        global g_time_limit
        run_process = subprocess.run(
            input_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            timeout=g_time_limit,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        res = ""
        if e.stdout is not None:
            if isinstance(e.stdout, bytes):
                res += e.stdout.decode("utf-8")
            else:
                res += e.stdout
        if e.stderr is not None:
            if isinstance(e.stderr, bytes):
                res += e.stderr.decode("utf-8")
            else:
                res += e.stderr
        return res
    return run_process.stdout + run_process.stderr


class HostCmdToolInput(BaseModel):
    """Input for the run_cmd_on_host tool."""

    cmds: List[str] = Field(
        description="A list of commands to be executed on the host, where each command should be a complete and executable statement",
    )


@tool(args_schema=HostCmdToolInput)
def run_cmd_on_host(cmds: List[str]) -> str:
    """Run the multiple commands on the host"""
    print("------------------------------tool called-------------------------")
    res = _run_cmd_on_host(cmds)
    print(f"run_cmd_on_host called with cmd: {cmds}. with result: {res}")
    print(f"\n")
    return res
