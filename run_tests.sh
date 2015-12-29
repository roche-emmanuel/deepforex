#!/bin/sh

# for i in 5 10 20 25 30 35 40 45 50; 
# for i in 25; 
# do
#   echo "Starting training for seq length = $i"
#   screen -dmS forex_seq_$i bash -c "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix seq_$i -seq_length $i"
# done

screen -dmS forex_v13 bash -c "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v13"
