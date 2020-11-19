# trading-bot
**trading-bot** is a command line tool that uses [FXCM Rest API](https://github.com/fxcm/RestAPI "FXCM Rest API") and Pytorch (through [tch-rs](https://github.com/LaurentMazare/tch-rs "tch-rs")) in order to train and apply a model (a simple Neural Network with a single hidden layer) for forex trading.

The following commands are supported :
* `fetch` - used for fetching historical data (i.e. candlesticks) for a specific forex instrument
* `train` - used for training a model using historical data
* `eval` - used for applying a model on historical data and evaluating its performance
* `trade` - used for opening a position for a specific forex instrument using a trained model
