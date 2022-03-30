for filename in history/*.csv; do
    [ -e "$filename" ] || continue
    for offline_method in dm ips dr; do
        echo $offline_method
        for i in {1..10}; do
	   python3 train_bandit.py mushroom epsilon_greedy mushroom_configs/greedy.yaml --offline=$filename --offline_method=$offline_method
	   #python3 train_bandit.py mushroom neural_ucb mushroom_configs/neuralucb.yaml --offline=$filename --offline_method=$offline_method
	   python3 train_bandit.py mushroom bayes_by_backprob mushroom_configs/bayes_by_backprob.yaml --offline=$filename --offline_method=$offline_method
   	done
    done
done

