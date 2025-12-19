"""Utils for logging."""
from hydra.core.hydra_config import HydraConfig
from logdir import LogDir


def symlink_last_run(logdir: LogDir):
    """Set up a symlink called last_run to point to this directory. Definitely
    not thread-safe."""
    last_run = logdir.logdir.parent / "last_run"
    last_run.unlink(missing_ok=True)
    last_run.symlink_to(logdir.logdir.name)


def setup_logdir_from_hydra() -> LogDir:
    """Creates a LogDir based on the current Hydra output directory."""
    logdir = LogDir(
        name=HydraConfig.get().job.name,
        custom_dir=HydraConfig.get().runtime.output_dir,
    )
    logdir.readme(date=True, git_commit=True)
    # Causes trouble when running in parallel.
    #  symlink_last_run(logdir)
    return logdir
