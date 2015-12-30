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

# screen -dmS forex_test bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v14 -train_size 1000"
# screen -r forex_test

# screen -dmS forex_test bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v15 -num_remas 1"
# screen -r forex_test

# screen -dmS forex_test bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v16 -num_remas 2"
# screen -r forex_test

# screen -dmS forex_test bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v16b -num_remas 2 -rnn_size 200"
# screen -r forex_test

# screen -dmS forex_v17 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v17 -num_remas 1 -seed 124"
# screen -r forex_v17

# screen -dmS forex_v18 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v18 -num_remas 1 -eval_size 50 -max_sessions 400 -seed 125"
# screen -r forex_v18

# screen -dmS forex_test bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v19 -num_remas 1 -optim cg"
# screen -r forex_test

# screen -dmS forex_v20 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v20 -num_remas 1 -batch_size 80 -max_epochs 15 -initial_max_epochs 50"
# screen -r forex_v20

screen -dmS forex_v21 bash -c \
  "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v21 -num_remas 1 -batch_size 80 -max_epochs 15 -initial_max_epochs 50"
screen -r forex_v21