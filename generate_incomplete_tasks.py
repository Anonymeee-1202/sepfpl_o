#!/usr/bin/env python3
import argparse
import datetime
import glob
import os
import shlex
from pathlib import Path


DEFAULT_ROOT = Path("/home/liuxin25/code/dp-fpl")
DEFAULT_TASK_LIST = DEFAULT_ROOT / "tasks" / "task_list.sh"
DEFAULT_LOGS_DIR = DEFAULT_ROOT / "logs"
DEFAULT_TASKS_DIR = DEFAULT_ROOT / "tasks"
COMPLETION_MARKERS = (
    "maximum test local acc",
    "训练完成",
    "Training finished",
)


def parse_task_commands(task_list_path: Path) -> list[str]:
    commands: list[str] = []
    with task_list_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith("bash srun_main.sh"):
                commands.append(stripped)
    return commands


def extract_command_metadata(command: str) -> dict:
    tokens = shlex.split(command)
    if len(tokens) < 9:
        raise ValueError(f"无法解析任务命令: {command}")

    dataset_config = Path(tokens[3])
    method = tokens[5]
    client_num = tokens[6]
    epsilon = tokens[7]
    seed = tokens[8]

    dataset = dataset_config.stem

    return {
        "command": command,
        "dataset": dataset,
        "method": method,
        "client_num": client_num,
        "epsilon": epsilon,
        "seed": seed,
    }


def list_related_logs(logs_dir: Path, metadata: dict) -> list[Path]:
    prefix = f"{metadata['client_num']}_{metadata['epsilon']}_{metadata['seed']}_"
    pattern = logs_dir / metadata["dataset"] / metadata["method"] / f"{prefix}*.log"
    paths = [Path(path) for path in glob.glob(str(pattern))]
    try:
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except FileNotFoundError:
        paths = [p for p in paths if p.exists()]
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def is_log_complete(log_path: Path) -> bool:
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
    except FileNotFoundError:
        return False

    if any(marker in content for marker in COMPLETION_MARKERS):
        return True
    if "Traceback (most recent call last)" in content or "ERROR" in content:
        return False
    return False


def find_incomplete_commands(
    commands: list[str],
    logs_dir: Path,
) -> list[str]:
    incomplete: list[str] = []
    for command in commands:
        metadata = extract_command_metadata(command)
        logs = list_related_logs(logs_dir, metadata)
        if not logs:
            incomplete.append(command)
            continue
        latest_log = logs[0]
        if is_log_complete(latest_log):
            continue
        incomplete.append(command)
    return incomplete


def resolve_output_path(tasks_dir: Path, output: str | None) -> Path:
    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = tasks_dir / output_path
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = tasks_dir / f"incomplete_tasks_{timestamp}.sh"
    return output_path


def write_shell_script(
    output_path: Path,
    commands: list[str],
    cuda_devices: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("#!/bin/bash\n")
        file.write(f"# Auto-generated on {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        file.write(f"# Total incomplete tasks: {len(commands)}\n\n")

        if cuda_devices:
            file.write(f"export CUDA_VISIBLE_DEVICES={cuda_devices}\n\n")

        if not commands:
            file.write("# 所有实验均已完成，无需重新运行。\n")
            return

        width = len(str(len(commands)))
        for idx, command in enumerate(commands, 1):
            file.write(f"# Task {idx:>{width}}/{len(commands)}\n")
            file.write(f"{command}\n\n")

    os.chmod(output_path, 0o755)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据日志检测未完成实验，并生成对应的运行脚本。"
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        default=DEFAULT_TASK_LIST,
        help="任务列表脚本路径",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="日志目录路径",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="输出脚本存放目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出脚本文件名（可选）",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="指定脚本执行时使用的 CUDA_VISIBLE_DEVICES 参数，例如 '0' 或 '0,1'。",
    )
    args = parser.parse_args()

    commands = parse_task_commands(args.task_list)
    if not commands:
        raise RuntimeError(f"未在 {args.task_list} 中找到任何任务命令。")

    incomplete_commands = find_incomplete_commands(commands, args.logs_dir)

    output_path = resolve_output_path(args.tasks_dir, args.output)
    write_shell_script(
        output_path,
        incomplete_commands,
        cuda_devices=args.cuda_devices,
    )

    print(f"总任务数: {len(commands)}")
    print(f"未完成任务数: {len(incomplete_commands)}")
    print(f"生成脚本: {output_path}")


if __name__ == "__main__":
    main()

