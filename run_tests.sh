#!/bin/sh

# for i in 5 10 20 25 30 35 40 45 50; 
# for i in 25; 
# do
#   echo "Starting training for seq length = $i"
#   screen -dmS forex_seq_$i bash -c "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix seq_$i -seq_length $i"
# done

# screen -dmS forex_test bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v13b"
# screen -r forex_test

screen -dmS forex_test bash -c \
  "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v14 -train_size 1000"
screen -r forex_test
