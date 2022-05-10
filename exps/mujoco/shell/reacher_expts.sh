python barl_run.py -m env_config_file=config/envs/barl_reacher.yaml agent=bptt name=hucrl_reacher_thompson exploration=thompson seed="range(5)" hydra/launcher=joblib &
python barl_run.py -m env_config_file=config/envs/barl_reacher.yaml agent=bptt name=hucrl_reacher_greedy exploration=greedy seed="range(5)" hydra/launcher=joblib &
python barl_run.py -m env_config_file=config/envs/barl_reacher.yaml agent=bptt name=hucrl_reacher exploration=optimistic seed="range(5)" hydra/launcher=joblib &
wait
