#Process that i think should be here:
# 1. Train the model until the loss is acceptable
# 2. "Test" the model with changing data - simulate long run of forecasting without traning (todo: make a changing data or find something to simulate drifing here)
# 3. When drift detected, deploy corrections - either wait, then run retraining, at the end reset the model
