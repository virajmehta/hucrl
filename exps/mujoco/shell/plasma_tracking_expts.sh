python barl_run.py env_config_file=config/envs/plasma_tracking.yaml agent=bptt name=hucrl_plasma_tracking_thompson exploration=thompson seed="range(5)" hydra/launcher=joblib
python barl_run.py env_config_file=config/envs/plasma_tracking.yaml agent=bptt name=hucrl_plasma_tracking_greedy exploration=greedy seed="range(5)" hydra/launcher=joblib
python barl_run.py env_config_file=config/envs/plasma_tracking.yaml agent=bptt name=hucrl_plasma_tracking exploration=optimistic seed="range(5)" hydra/launcher=joblib
