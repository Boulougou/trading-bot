#!/bin/sh  
while true  
do  
    #cargo run -- trade -m ./models/EUR_USD_Min1_120_30/ -a 1 --min-profit 1.0 --max-profit 1.1 --max-loss 20.0 --max-used-margin 0.01
    #cargo run -- trade -m ./models/EUR_USD_Min30_48_1_linear_agg_25/ -a 1 --max-loss 1.0 --max-used-margin 0.003
    cargo run -- trade -m ./models/EUR_USD_Min30_96_1_nn16_def_25/ -a 1 --max-loss 20.0 --max-used-margin 0.003
    date
    echo "=======================================================\n"
    sleep 1740
done
