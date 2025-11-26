"""Background Agent System Core Components."""

from .agent import BackgroundAgent, AgentConfig, AgentPriority
from .task_queue import TaskQueue, Task, TaskPriority
from .agent_system import AgentSystem

try:
    from .file_monitor import FileMonitor
    __all__ = [
        "BackgroundAgent",
        "AgentConfig",
        "AgentPriority",
        "TaskQueue",
        "Task",
        "TaskPriority",
        "AgentSystem",
        "FileMonitor",
    ]
except ImportError:
    __all__ = [
        "BackgroundAgent",
        "AgentConfig",
        "AgentPriority",
        "TaskQueue",
        "Task",
        "TaskPriority",
        "AgentSystem",
    ]

