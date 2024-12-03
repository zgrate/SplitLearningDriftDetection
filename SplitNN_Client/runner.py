import os
import threading
from datetime import datetime
from time import sleep

from SplitNN_Client.client import run_client
from SplitNN_Client.server_connection import ServerConnection

clients = int(input("Number of clients "))

date = input("Do you want to reset server side data? Y/N " )
if date.lower() == "y":
    print("Resetting server side data")
    ServerConnection("null").reset_runner()
else:
    print("not resseting")

print("Starting runner in 5 seconds")
sleep(5)

folder = datetime.now().strftime("%Y%m%d%H%M")
os.mkdir(folder)

for i in range(clients):
    print("Starting Client", i)
    threading.Thread(target=run_client, args=[i, clients, folder]).start()
