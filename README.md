# trading-bot
**trading-bot** is a command line tool that uses [FXCM Rest API](https://github.com/fxcm/RestAPI "FXCM Rest API") and Pytorch (through [tch-rs](https://github.com/LaurentMazare/tch-rs "tch-rs")) in order to train and apply a model (a simple Neural Network with a single hidden layer) for forex trading.

The following commands are supported :
* `fetch` - used for fetching historical data (i.e. candlesticks) for a specific forex instrument
* `train` - used for training a model using historical data
* `eval` - used for applying a model on historical data and evaluating its performance
* `trade` - used for opening a position for a specific forex instrument using a trained model

## Examples

### Fetching historical data
The command below will fetch candlesticks of one minute (`-t Min1`) for symbol EUR/USD (`-s EUR/USD`) between the specified dates:
```
cargo run -- fetch -t Min1 -s EUR/USD --from-date 201009010000 --to-date 201101010000
```
The data will be written inside directory `./history/EUR_USD_Min1_201009010000_201101010000/`

### Training model
After fetching historical data, it is possible to train the model using the `train` command:
```
cargo run -- train --input ./history/EUR_USD_Min1_201009010000_201101010000/ --input-window 30 --pred-window 5 --learning-rate 1e-3 --learning-iterations 3000 -m EUR_USD_Min1_30_5
```
The command above will train a model that based on last 30 minutes (`--input-window 30`), it will predict the minimum and maximum bid price of EUR/USD on the next 5 minutes (`--pred-window 5`). The model will be stored under `./models/EUR_USD_Min1_30_5`
