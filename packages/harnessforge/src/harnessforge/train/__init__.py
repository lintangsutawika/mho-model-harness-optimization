"""Training on platoon through a miles-tinker Tinker API gateway (workstream E).

    harnessforge.train.config    TrainRun: one run's settings (JSON), old-flag mapping
    harnessforge.train.gateway   the miles-tinker gateway job (Ray over 1 trainer + R rollout nodes)
    harnessforge.train.platoon   the `platoon train` config for a run
    harnessforge.train.driver    the client job: platoon train + HF exporter, stops the gateway
    harnessforge.train.export    HF checkpoints from the disk-delta weight stream
    python -m harnessforge.train render|launch|status|stop|export
"""
