#!/bin/bash

check_error() {
    if [ $? -ne 0 ]; then
        echo "Error encountered in $1. Exiting."
        exit 1
    fi
}

emotions = ('realization' 'fear' 'annoyance' 'admiration' 'relief' 'confusion' 'approval' 'remorse' 'sadness' 'grief' 'nervousness' 'optimism' 'disgust' 'joy' 'amusement' 'embarrassment' 'love' 'pride' 'gratitude' 'disappointment' 'surprise' 'desire' 'anger' 'curiosity' 'disapproval' 'excitement' 'caring' 'neutral')

num_gens=3
output_name="output"

for emotion in "${emotions[@]}"; do
	python3 fine_tune_initial.py $emotion
	check_error "fine_tune_initial.py for emotion $emotion"

	python3 poll_jobs.py
	check_error "poll_jobs.py for emotion $emotion, gen=1"

	python3 run_initial.py $emotion $output_name
	check_error "run_initial.py emotion = $emotion"

    for gen in $(seq 2 $num_gens); do
        python3 fine_tune_multistep.py $emotion $gen
		check_error "fine_tune_multistep.py for generation $gen for emotion $emotion"

        python3 poll_jobs.py
		check_error "poll_jobs.py for emotion $emotion, gen=$gen"

        python3 run_step.py $emotion $gen $output_name
		check_error "run_step.py for generation $gen for emotion $emotion"
    done

	python3 analyze.py $emotion $output_name
done