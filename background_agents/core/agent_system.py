"""Main agent system orchestrator."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .agent import BackgroundAgent, AgentConfig, AgentPriority
from .task_queue import TaskQueue, Task, TaskStatus, TaskPriority
import subprocess

try:
    from .file_monitor import FileMonitor
    FILE_MONITOR_AVAILABLE = True
except ImportError:
    FILE_MONITOR_AVAILABLE = False
    FileMonitor = None


class AgentSystem:
    """Main system that manages all background agents."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()

        self.task_queue = TaskQueue(Path(self.config['database']['path']))
        self.agents: Dict[str, BackgroundAgent] = {}
        self._running = False
        self.logger = logging.getLogger("agent_system")
        self.file_monitor = None
        self._processed_files_path = Path("background_agents/.processed_files.json")
        self._processed_files: set = self._load_processed_files()
        self.activity_digest: Optional[Dict] = None

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config['logging']
        log_dir = Path(log_config['directory'])
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config['system']['log_level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_dir / 'system.log'),
                logging.StreamHandler()
            ]
        )

    def register_agent(self, agent: BackgroundAgent) -> None:
        """
        Register an agent with the system.

        Args:
            agent: The agent to register
        """
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")

    def _load_processed_files(self) -> set:
        """Load set of already processed files."""
        if not self._processed_files_path.exists():
            return set()

        try:
            with open(self._processed_files_path) as f:
                data = json.load(f)
                return set(data.get('files', []))
        except Exception as e:
            self.logger.error(f"Error loading processed files: {e}")
            return set()

    def _save_processed_files(self):
        """Save set of processed files."""
        try:
            self._processed_files_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._processed_files_path, 'w') as f:
                json.dump({'files': list(self._processed_files)}, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving processed files: {e}")

    def _on_file_detected(self, file_path: Path):
        """Callback when a new file is detected."""
        # Check if already processed
        file_str = str(file_path.absolute())
        if file_str in self._processed_files:
            self.logger.debug(f"File already processed: {file_path.name}")
            return

        # Create task for document monitor
        if 'document_monitor' not in self.agents:
            self.logger.warning("document_monitor agent not registered")
            return

        self.logger.info(f"Creating task for new file: {file_path.name}")

        task = Task.create(
            agent_name='document_monitor',
            task_type='process_file',
            data={'file_path': str(file_path)},
            priority=TaskPriority.HIGH
        )

        self.task_queue.add_task(task)

        # Mark as processed
        self._processed_files.add(file_str)
        self._save_processed_files()

    def _setup_file_monitoring(self):
        """Setup file system monitoring for configured directories."""
        if not FILE_MONITOR_AVAILABLE:
            self.logger.warning("watchdog not available - file monitoring disabled")
            self.logger.info("Install with: pip install watchdog")
            return

        doc_agent_cfg = self.config['agents'].get('document_monitor', {})
        if not doc_agent_cfg.get('enabled', False):
            self.logger.info("Document monitor disabled; skipping directory monitoring")
            return

        monitored_dirs = self.config.get('monitored_directories', [])
        if not monitored_dirs:
            self.logger.info("No directories configured for monitoring")
            return

        self.file_monitor = FileMonitor()

        for dir_config in monitored_dirs:
            dir_path = Path(dir_config['path'])
            extensions = dir_config.get('watch_extensions', ['.pdf'])
            recursive = dir_config.get('recursive', True)
            auto_process = dir_config.get('auto_process', True)

            if not auto_process:
                continue

            if not dir_path.exists():
                self.logger.warning(f"Directory does not exist: {dir_path}")
                continue

            # Watch for new files
            self.file_monitor.watch_directory(
                dir_path,
                extensions,
                self._on_file_detected,
                recursive=recursive
            )

            # Optionally process a few existing files on startup
            self.logger.info(f"Scanning for existing files in: {dir_path}")
            self.file_monitor.scan_existing_files(
                dir_path,
                extensions,
                self._on_file_detected,
                recursive=recursive,
                max_files=5  # Only process 5 existing files initially
            )

        # Start monitoring
        if self.file_monitor and self.file_monitor.handlers:
            self.file_monitor.start()
        else:
            self.file_monitor = None

    async def initialize_agents(self) -> None:
        """Initialize all registered agents."""
        for agent in self.agents.values():
            try:
                await agent.initialize()
                # Attach the latest activity digest so all agents start with context
                if self.activity_digest is None:
                    self.activity_digest = self._load_activity_digest()
                try:
                    # Provide digest under a common attribute if available
                    setattr(agent, "activity_digest", self.activity_digest)
                except Exception:
                    pass
            except Exception as e:
                self.logger.error(f"Failed to initialize {agent.name}: {e}")

    async def shutdown_agents(self) -> None:
        """Shutdown all agents."""
        for agent in self.agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.error(f"Failed to shutdown {agent.name}: {e}")

    async def process_agent_tasks(self, agent_name: str) -> None:
        """
        Continuously process tasks for a specific agent.

        Args:
            agent_name: Name of the agent
        """
        agent = self.agents[agent_name]

        while self._running:
            try:
                # Get next task
                task = self.task_queue.get_next_task(agent_name)

                if task is None:
                    # No tasks, sleep briefly
                    await asyncio.sleep(1)
                    continue

                # Update status to in progress
                self.task_queue.update_task_status(task.id, TaskStatus.IN_PROGRESS)

                # Process task
                try:
                    result = await agent.run_task(task.data)

                    # Mark as completed
                    self.task_queue.update_task_status(
                        task.id,
                        TaskStatus.COMPLETED,
                        result={'output': result}
                    )

                except Exception as e:
                    self.logger.error(f"Task {task.id} failed: {e}")

                    # Try to retry or mark as failed
                    if not self.task_queue.increment_retry(task.id):
                        self.task_queue.update_task_status(
                            task.id,
                            TaskStatus.FAILED,
                            error=str(e)
                        )

            except Exception as e:
                self.logger.error(f"Error in agent {agent_name} task loop: {e}")
                await asyncio.sleep(5)

    async def schedule_periodic_tasks(self) -> None:
        """Schedule periodic tasks for agents based on their intervals."""
        # Track last run time for each agent
        last_run = {}

        while self._running:
            try:
                current_time = asyncio.get_event_loop().time()

                for agent_name, agent_config in self.config['agents'].items():
                    if not agent_config['enabled']:
                        continue

                    # Determine interval
                    if 'interval_minutes' in agent_config and agent_config['interval_minutes']:
                        interval_seconds = agent_config['interval_minutes'] * 60
                    elif 'interval_hours' in agent_config and agent_config['interval_hours']:
                        interval_seconds = agent_config['interval_hours'] * 3600
                    else:
                        continue  # On-demand only

                    # Check if enough time has passed since last run
                    if agent_name in last_run:
                        time_since_last = current_time - last_run[agent_name]
                        if time_since_last < interval_seconds:
                            continue  # Not time yet

                    # Create task with proper data for each agent type
                    task_data = self._create_agent_task_data(agent_name, agent_config)
                    if task_data is None:
                        continue  # Skip if we can't create proper data

                    task = Task.create(
                        agent_name=agent_name,
                        task_type='periodic',
                        data=task_data,
                        priority=TaskPriority[agent_config['priority'].upper()]
                    )

                    self.task_queue.add_task(task)
                    last_run[agent_name] = current_time
                    self.logger.info(f"Scheduled task for {agent_name}")

                # Check every 30 seconds
                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"Error in scheduler: {e}", exc_info=True)
                await asyncio.sleep(30)

    def _create_agent_task_data(self, agent_name: str, agent_config: dict) -> Optional[Dict]:
        """Create appropriate task data for each agent type."""
        if agent_name == 'project_organizer':
            # Project organizer gets configuration options
            return {
                'root': agent_config.get('root_directory', str(self.config_path.parent)),
                'mode': 'organize',
                'enable_file_naming_standardization': agent_config.get('enable_file_naming_standardization', True),
                'enable_folder_organization': agent_config.get('enable_folder_organization', True),
                'enable_cleanup_empty_folders': agent_config.get('enable_cleanup_empty_folders', True),
                'enable_duplicate_detection': agent_config.get('enable_duplicate_detection', True)
            }

        elif agent_name == 'document_monitor':
            # Document monitor will be triggered by file watcher, not scheduler
            return None

        elif agent_name == 'legal_research':
            # Research agent needs cases - get from processed documents
            cases = self._get_recent_cases()
            if not cases:
                self.logger.debug("No cases available for research agent")
                return None
            return {
                'research_type': 'summary',
                'cases': cases[:10]  # Limit to 10 most recent
            }

        elif agent_name == 'citation_network':
            # Citation network needs case list
            cases = self._get_all_cases()
            if len(cases) < 10:
                self.logger.debug("Not enough cases for citation network")
                return None
            return {
                'cases': cases
            }

        elif agent_name == 'pattern_detection':
            # Pattern detection needs multiple cases
            cases = self._get_all_cases()
            if len(cases) < 5:
                self.logger.debug("Not enough cases for pattern detection")
                return None
            return {
                'cases': cases
            }

        elif agent_name == 'settlement_optimizer':
            # Settlement optimizer needs specific parameters
            # For now, run with default parameters
            return {
                'case_name': 'default',
                'success_probability': 0.65,
                'damages_mean': 1000000,
                'damages_std': 200000,
                'legal_costs': 75000,
                'risk_aversion': 0.5,
                'iterations': 10000
            }

        return None

    def _get_recent_cases(self) -> List[Dict]:
        """Get recently processed cases from output directory."""
        cases = []
        doc_dir = Path("background_agents/outputs/document_analysis")

        if not doc_dir.exists():
            return cases

        # Get recent JSON files
        json_files = sorted(doc_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

        for json_file in json_files[:20]:  # Last 20 documents
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    extracted = data.get('extracted_data', {})
                    if extracted:
                        cases.append(extracted)
            except Exception as e:
                self.logger.error(f"Error reading {json_file}: {e}")

        return cases

    def _get_all_cases(self) -> List[Dict]:
        """Get all processed cases."""
        cases = []
        doc_dir = Path("background_agents/outputs/document_analysis")

        if not doc_dir.exists():
            return cases

        for json_file in doc_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    extracted = data.get('extracted_data', {})
                    if extracted:
                        cases.append(extracted)
            except Exception as e:
                self.logger.error(f"Error reading {json_file}: {e}")

        return cases

    async def start(self) -> None:
        """Start the agent system."""
        self.logger.info("Starting agent system...")
        self._running = True

        # Generate or refresh shared activity digest before agents begin work
        try:
            self._generate_activity_digest(days=7)
        except Exception as e:
            self.logger.warning(f"Could not generate activity digest on start: {e}")

        # Initialize agents
        await self.initialize_agents()

        # Setup file monitoring
        self._setup_file_monitoring()

        # Start task processors for each agent
        tasks = []
        for agent_name in self.agents.keys():
            tasks.append(asyncio.create_task(self.process_agent_tasks(agent_name)))

        # Start scheduler
        tasks.append(asyncio.create_task(self.schedule_periodic_tasks()))

        # Run until stopped
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Agent system cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the agent system."""
        self.logger.info("Stopping agent system...")
        self._running = False

        # Stop file monitoring
        if self.file_monitor:
            self.file_monitor.stop()

        # Save processed files
        self._save_processed_files()

        # Shutdown agents
        await self.shutdown_agents()

        # Update digest on clean shutdown so latest outputs are captured
        try:
            self._generate_activity_digest(days=7)
        except Exception as e:
            self.logger.warning(f"Could not generate activity digest on stop: {e}")

        self.logger.info("Agent system stopped")

    def get_system_stats(self) -> Dict:
        """Get overall system statistics."""
        return {
            'running': self._running,
            'agents': {
                name: agent.get_stats()
                for name, agent in self.agents.items()
            },
            'queue': self.task_queue.get_stats()
        }

    # ---------------- Internal helpers for Activity Digest -----------------

    def _generate_activity_digest(self, days: int = 7) -> None:
        """Invoke the recent_activity_digest script to update shared reports."""
        md_out = Path("reports/analysis_outputs/activity_digest.md")
        json_out = Path("reports/analysis_outputs/activity_digest.json")
        md_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python3",
            "scripts/recent_activity_digest.py",
            "--days", str(days),
            "--output-md", str(md_out),
            "--output-json", str(json_out),
        ]
        self.logger.info("Refreshing Activity Digest for agents...")
        subprocess.run(cmd, check=False)
        # Reload digest cache
        self.activity_digest = self._load_activity_digest()

    def _load_activity_digest(self) -> Optional[Dict]:
        """Load the most recent digest JSON if available."""
        try:
            path = Path("reports/analysis_outputs/activity_digest.json")
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load activity digest: {e}")
            return None
