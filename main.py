# Hi,
# great thanks (although i think in the long term the Flower or adjusting the Feddrift would have been the better choice because we could have compared it easily with federated learning then but it is as it is). How did you test your code? Can you do the following:
# - use a dataset (like the MINST that you are using), give each client different data, let it train and verify that each client has its own model and the server has a common model. Vary the number of clients from 1 to say 10.
# - adjust the program so that we can have different ratios of split client:server, like client 2 layers, server 4 layers, client 3 layers, server 3 layers, client 4 layers, server 2 layers, etc,
# - for evaluation record different performance indicators as in Feddrift and additionally record training time, inference time, amount of network traffic in total for training and for inference.
# - compare it with a centralized model (the standalone in the Feddrift repo)
# - use different models and datasets (see Feddrift)
# - change batch size, client optimizer (adam, etc...), learning rarte, ..weight decay, ..epochs
#
# For dataset, try different ones: mnist, shakespear, stackoverflow, ... check Feddrift repo
#
# Then we need a strategy for the drift detection and adaptation but that we do only once we know that the split learning works.
