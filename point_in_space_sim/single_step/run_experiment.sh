#!/bin/bash

check_error() {
    if [ $? -ne 0 ]; then
        echo "Error encountered in $1. Exiting."
        exit 1
    fi
}
emotions = ('realization' 'fear' 'annoyance' 'admiration' 'relief' 'confusion' 'approval' 'remorse' 'sadness' 'grief' 'nervousness' 'optimism' 'disgust' 'joy' 'amusement' 'embarrassment' 'love' 'pride' 'gratitude' 'disappointment' 'surprise' 'desire' 'anger' 'curiosity' 'disapproval' 'excitement' 'caring' 'neutral')

for emotion in "${emotions[@]}"; do
	python3 fine_tune.py $emotion
	check_error "fine_tune.py for emotion $emotion"

	python3 poll_jobs.py
	check_error "poll_jobs.py for emotion $emotion"

	python3 run.py $emotion
	check_error "run.py emotion = $emotion"

	python3 analyze.py $emotion
done