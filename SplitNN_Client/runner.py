import os
import signal
import threading
from datetime import datetime
from time import sleep

from SplitNN_Client.client import run_client
from SplitNN_Client.server_connection import ServerConnection


class RunnerThread():

    def __init__(self):
        self.global_stop = False

    def start_runner(self):
        clients = int(input("Number of clients "))

        date = input("Do you want to reset server side data? Y/N ")
        sync_mode = input("Synchronizer mode? Y/n").lower() == "y"
        if date.lower() == "y":
            print("Resetting server side data")
            ServerConnection("null").reset_runner()
        else:
            print("not resseting")

        print("Starting runner in 5 seconds")
        sleep(5)

        ServerConnection("null").prepare_runner(clients)

        folder = datetime.now().strftime("%Y%m%d%H%M%s")
        os.mkdir(folder)

        threads = []

        for i in range(clients):
            print("Starting Client", i)
            t = threading.Thread(target=run_client, args=[i, clients, folder, self, sync_mode])
            threads.append(t)
            t.start()

        def handler(signum, frame):
            print(signum)
            self.global_stop = True

        signal.signal(signal.SIGINT, handler)

        while not self.global_stop:
            try:
                sleep(1)
            except KeyboardInterrupt:
                print("STOP!")
                self.global_stop = True

        print("Stopping runners")
        while any(t.is_alive() for t in threads):
            sleep(1)

    def get_global_stop(self):
        return self.global_stop


if __name__ == "__main__":
    RunnerThread().start_runner()
