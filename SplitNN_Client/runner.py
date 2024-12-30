import os
import signal
import threading
from datetime import datetime
from time import sleep

from SplitNN_Client.client import run_client, ClientModel, TrainingSuite
from SplitNN_Client.server_connection import ServerConnection


class SplitLearningRunner:

    def __init__(self, all_props_dict, clients, client_learning_rate, *, server_learning_rate=0.001, sync_mode=False, seconds_running=0, client_epochs_limit=0, target_loss=0, all_client_loss=False, **kwargs):
        self.all_props_dict = all_props_dict
        self.global_stop = False
        self.clients = clients
        self.client_epochs_limit = client_epochs_limit
        self.seconds_running = seconds_running
        self.sync_mode = sync_mode
        self.folder = datetime.now().strftime("%Y%m%d%H%M%s")
        self.server_connection = ServerConnection("runner")
        self.target_loss = target_loss
        self.all_clients_loss = all_client_loss
        self.client_losses = {}
        self.client_learning_rate = client_learning_rate
        self.server_learning_rate = server_learning_rate
        self.testing = True

    def client_response(self, client_id, data):
        self.client_losses[client_id] = data['loss']
        if self.all_clients_loss:
            if all([x <= self.target_loss for x in self.client_losses.values()]):
                print("All loses are < los target")
                self.global_stop = True

        else:
            if self.target_loss > 0 and data['loss'] <= self.target_loss:
                print(f"We have a target loss for client! {client_id}")
                self.global_stop = True

    def start_runner(self):
        print("Starting runner in 5 seconds")
        sleep(5)
        self.server_connection.prepare_runner(self.all_props_dict)

        os.mkdir(self.folder)

        threads = []

        for i in range(self.clients):
            print("Starting Client", i)
            t = threading.Thread(target=run_client, args=[i, self])
            threads.append(t)
            t.start()
            sleep(0.5)

        def handler(signum, frame):
            print(signum)
            self.global_stop = True

        signal.signal(signal.SIGINT, handler)

        seconds_counter = 0
        while not self.global_stop:
            try:
                sleep(1)
                if self.seconds_running > 0:
                    seconds_counter += 1
                    if seconds_counter % 15 == 0:
                        print("Passed", seconds_counter)
                    if seconds_counter >= self.seconds_running:
                        self.global_stop = True
                        print("STOP COUNTER REACHED TARGET TIME!!")

            except KeyboardInterrupt:
                print("STOP!")
                self.global_stop = True


        print("Stopping runners")

        sample_client = TrainingSuite(None, None, learning_rate=self.client_learning_rate)

        if not self.server_connection.save_report({
            "client_model": str(sample_client.model.model),
            "client_optimizer": str(sample_client.optimizer),
            "clients": self.clients,
            "sync_mode": self.sync_mode,
            "seconds_running": self.seconds_running,
            "client_folder": self.folder,
            "target_loss": self.target_loss,
            "client_learning_rate": self.client_learning_rate,
            "server_learning_rate": self.server_learning_rate,
        }):
            print("KURWA")

        # while any(t.is_alive() for t in threads):
        #     sleep(1)

    def get_global_stop(self):
        return self.global_stop


if __name__ == "__main__":

    if True:
        default = {
            "clients": 5,
            "sync_mode": False,
            "seconds_running": 0,
            "client_epochs_limit": 0,
            "target_loss": 1,
            "all_client_loss": True,
            "server_optimiser_options": {
                "lr": 0.001,
            },
            "client_learning_rate": 0.001,
            "server_load_save_data": None,
            "client_load_directory": None,
            "load_only": False,
            "reset_logs": True,
            "reset_nn": True,
            "mode": "train" #'train', 'predict_random'
        }
        clients = 5
        server_load_save_data = "server_test_1"
        client_load_directory = "client_test_1"

        params = {
            "clients": clients,
            'server_load_save_data': server_load_save_data,
            'client_load_directory': client_load_directory
        }

        settings = [
            # {"clients": 2, **default},
            {**default, **params, 'target_loss': 0.2, 'reset_logs': True, 'reset_nn': True, "mode": "train"},
            {**default, **params, 'reset_logs': False, 'reset_nn': False, "load_only": True, "mode": "predict_random"},
            # {"clients": 4, **default},
            # {"clients": 5, **default},
            # {"clients": 6, **default},
            # {"clients": 7, **default},
            # {"clients": 8, **default},
            # {"clients": 9, **default},
            # {"clients": 10, **default},
        ]
        for setting in settings:
            print("Running", setting)
            SplitLearningRunner(setting, **setting).start_runner()
            sleep(30)
        exit(0)

    setting = {
        "clients": 5,
        "sync_mode": False,
        "seconds_running": 0,
        "client_epochs_limit": 0,
        "target_loss": 0.5,
        "all_client_loss": True,
        "client_learning_rate": 0.001,
        "server_learning_rate": 0.001,
    }
    print("Running runner with settings")
    print(setting)
    SplitLearningRunner(**setting).start_runner()
