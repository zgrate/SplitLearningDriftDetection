
Cut model

Initial Training on Client Side: Each client trains its portion of the model (up to the cut layer) using its local data. This step generates what is referred to as "smashed data," which represents intermediate outputs of the mode

Server Model Training

Client Model Update



Overview of Split Learning Steps
Split Learning is a collaborative machine learning approach that allows multiple clients to train a model without sharing their raw data. Instead, the learning process is divided between client and server models, enhancing privacy and reducing data transmission. Here are the key steps involved in implementing Split Learning:
1. Model Partitioning

Define the Cut Layer: The neural network is divided into two parts at a specific layer known as the cut layer. The first part is the client model, which processes the raw data, and the second part is the server model, which completes the training using outputs from the client

2. Client Model Training

Initial Training on Client Side: Each client trains its portion of the model (up to the cut layer) using its local data. This step generates what is referred to as "smashed data," which represents intermediate outputs of the model2

Data Transmission: The client sends only the smashed data (and potentially labels, depending on the configuration) to the server. This minimizes data exposure while allowing effective training

3. Server Model Training

Receive Smashed Data: The server receives the smashed data from one or more clients. It then uses this data to make predictions and compute loss values based on its portion of the model2

Backpropagation: The server computes gradients based on its predictions and sends these gradients back to the client for further training

4. Client Model Update

Gradient Application: Upon receiving gradients from the server, each client updates its model parameters accordingly. This step involves adjusting weights based on the received gradients to improve future predictions2

5. Iteration

Repeat Process: Steps 2 through 4 are repeated iteratively until convergence is achieved or a predetermined number of training epochs is completed. Each iteration allows for improved model accuracy while maintaining privacy since raw data is never shared1

6. Final Model Integration

Model Aggregation: If multiple clients are involved, their updated models can be aggregated at the server level to create a global model that benefits from diverse datasets without compromising individual data privacy3


# Question: Should server and client have the same Neural Networks? Should parts of it be the same?
