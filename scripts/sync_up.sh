# qrsh -l tmem=4G,gpu=true,h_rt=3600 -pe gpu 1
rsync -ruP cs-cluster:/home/zongchen/CBQ/results/sensitivity/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/sensitivity
