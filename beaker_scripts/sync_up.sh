# qrsh -l tmem=4G, h_rt=3600
rsync -ruP cs-cluster:/home/zongchen/CBQ/results/sensitivity_conjugate/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/sensitivity_conjugate
rsync -ruP cs-cluster:/home/zongchen/CBQ/results/finance/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/finance
rsync -ruP cs-cluster:/home/zongchen/CBQ/results/finance_stein/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/finance_stein/
rsync -ruP cs-cluster:/home/zongchen/CBQ/results/SIR/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/SIR

rsync -ruP myriad:/home/ucabzc9/Scratch/CBQ/results/sensitivity_conjugate/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/sensitivity_conjugate
rsync -ruP myriad:/home/ucabzc9/Scratch/CBQ/results/finance/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/finance
rsync -ruP myriad:/home/ucabzc9/Scratch/CBQ/results/finance_stein/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/finance_stein/
rsync -ruP myriad:/home/ucabzc9/Scratch/CBQ/results/SIR/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/results_server/SIR
rsync -ruP myriad:/home/ucabzc9/Scratch/CBQ/ablations/ /Users/hudsonchen/research/fx_bayesian_quaduature/CBQ/ablations/
