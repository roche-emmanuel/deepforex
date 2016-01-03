#!/bin/bash

source ~/scripts/profile.sh

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

# screen -dmS forex_v21 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v21 -num_remas 1 -batch_size 80 -max_epochs 15 -initial_max_epochs 50"
# screen -r forex_v21

# screen -dmS forex_v22 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v22 -num_remas 2 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -rnn_size 160"
# screen -r forex_v22

# screen -dmS forex_v23 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v23 -num_remas 1 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -rnn_size 160"
# screen -r forex_v23

# screen -dmS forex_v24 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v24 -num_remas 1 -batch_size 100 -max_epochs 15 -initial_max_epochs 100 -seq_length 50"
# screen -r forex_v24

# screen -dmS forex_v25 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v25 -num_remas 1 -num_emas 1 -batch_size 100 -max_epochs 15 -initial_max_epochs 100"
# screen -r forex_v25

# screen -dmS forex_v26 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v26 -num_remas 1 -rsi_period 9 -batch_size 100 -max_epochs 15 -initial_max_epochs 100"
# screen -r forex_v26

# here we want to forcast USDPJY log returns
# inputs are "EURUSD","AUDUSD","GBPUSD","NZDUSD","USDCAD","USDCHF","USDJPY"
# plus one REMA per price, so the target index should be: 13
# screen -dmS forex_v27 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v27 -num_remas 1 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 13"
# screen -r forex_v27

# we still want to predict USDPJY, but this time we have 5 cols per price, thus
# forcast_index = 6*5+1 = 31
# screen -dmS forex_v28 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v28 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 31"
# screen -r forex_v28

# screen -dmS forex_v29 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v29 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 1"
# screen -r forex_v29

# screen -dmS forex_v29b bash -c \
  # "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v29b -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 1 -seed 124"
# screen -r forex_v29b

# screen -dmS forex_v29c bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v29c -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 1 -seed 125"
# screen -r forex_v29c

# screen -dmS forex_v29d bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v29d -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 1 -seed 126"
# screen -r forex_v29d

# screen -dmS forex_v29e bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v29e -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 1 -seed 127"
# screen -r forex_v29e

# screen -dmS forex_v30 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v30 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 1"
# screen -r forex_v30

# screen -dmS forex_v29t3 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v29t3 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -forcast_index 1 -max_sessions 3"
# screen -r forex_v29t3

# Forcast the log return with offset = 3
# screen -dmS forex_v31 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v31 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 2"
# screen -r forex_v31

# screen -dmS forex_v32 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -suffix v32 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 1"
# screen -r forex_v32

# screen -dmS forex_v33 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -data_dir \"inputs/raw_2015_01_to_2015_10\" -suffix v33 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 1"
# screen -r forex_v33

# screen -dmS forex_v34 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -data_dir \"inputs/raw_2015_01_to_2015_10\" -suffix v34 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 2"
# screen -r forex_v34

# screen -dmS forex_v35a bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -data_dir \"inputs/raw_2015_01_to_2015_10\" -suffix v35a -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -eval_size 20 -max_sessions 1000 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 1 -seed 123"
# screen -r forex_v35a

# screen -dmS forex_v35b bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -data_dir \"inputs/raw_2015_01_to_2015_10\" -suffix v35b -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -eval_size 20 -max_sessions 1000 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 1 -seed 124"
# screen -r forex_v35b

# screen -dmS forex_v35c bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -data_dir \"inputs/raw_2015_01_to_2015_10\" -suffix v35c -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -eval_size 20 -max_sessions 1000 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 1 -seed 125"
# screen -r forex_v35c

# screen -dmS forex_v35d bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -data_dir \"inputs/raw_2015_01_to_2015_10\" -suffix v35d -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -eval_size 20 -max_sessions 1000 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 1 -seed 126"
# screen -r forex_v35d

# screen -dmS forex_v36 bash -c \
#   "source /home/kenshin/scripts/profile.sh; dforex_online_train -data_dir \"inputs/mt5_2015_12\" -suffix v36 -num_remas 2 -num_emas 1 -rsi_period 9 -batch_size 80 -eval_size 10 -max_sessions 1000 -max_epochs 15 -initial_max_epochs 100 -log_return_offsets 3 -forcast_index 1 -seed 123"
# screen -r forex_v36

# dforex_screen v36 \
#   -data_dir inputs/mt5_2015_12 \
#   -num_remas 2 \
#   -num_emas 1 \
#   -rsi_period 9 \
#   -batch_size 80 \
#   -eval_size 10 \
#   -max_sessions 1000 \
#   -max_epochs 15 \
#   -initial_max_epochs 100 \
#   -log_return_offsets 3 \
#   -forcast_index 1 \
#   -seed 123 \
#   -with_timetag 1 \
#   -with_close_only 1

dforex_screen_d v36b \
  -data_dir inputs/mt5_2015_12 \
  -num_remas 2 \
  -num_emas 1 \
  -rsi_period 9 \
  -batch_size 80 \
  -eval_size 10 \
  -max_sessions 1000 \
  -max_epochs 15 \
  -initial_max_epochs 100 \
  -log_return_offsets 3 \
  -forcast_index 1 \
  -seed 124 \
  -with_timetag 1 \
  -with_close_only 1

dforex_screen_d v36c \
  -data_dir inputs/mt5_2015_12 \
  -num_remas 2 \
  -num_emas 1 \
  -rsi_period 9 \
  -batch_size 80 \
  -eval_size 10 \
  -max_sessions 1000 \
  -max_epochs 15 \
  -initial_max_epochs 100 \
  -log_return_offsets 3 \
  -forcast_index 1 \
  -seed 125 \
  -with_timetag 1 \
  -with_close_only 1
