python barl_run.py env_config_file=config/envs/beta_tracking.yaml agent=bptt name=hucrl_beta_tracking_thompson exploration=thompson seed="range(5)" hydra/launcher=joblib
python barl_run.py env_config_file=config/envs/beta_tracking.yaml agent=bptt name=hucrl_beta_tracking_greedy exploration=greedy seed="range(5)" hydra/launcher=joblib
python barl_run.py env_config_file=config/envs/beta_tracking.yaml agent=bptt name=hucrl_beta_tracking exploration=optimistic seed="range(5)" hydra/launcher=joblib
