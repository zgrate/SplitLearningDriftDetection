from datetime import datetime


class ClientDataLogger():
    def __init__(self, client_id, client_file=None):
        self.client_id = client_id
        if client_file is None:
            self.client_file = f"client_file_{client_id}.csv"
        else:
            self.client_file = client_file

    def __enter__(self):
        self.client = open(self.client_file, "w+", encoding="utf-8")
        return self

    def __exit__(self):
        self.client.close()

    def write(self, s):
        print("Client",self.client_id,":", s, end="")
        self.client.write(s)

    def log_client_parameters(self, loss, acc, leng):
        self.write(f"P\t{datetime.now()}\t{self.client_id}\t{loss}\t{acc}\t{leng}\n")

    def log_client_transfer(self, data_lenght):
        self.write(f"T\t{datetime.now()}\t{self.client_id}\t{data_lenght}\n")

    def log_client_message(self, message):
        self.write(f"M\t{datetime.now()}\t{self.client_id}\t{message}\n")


class ServerDataLogger:

    def __init__(self, server_file="server_log.csv"):
        self.server_file = server_file

    def __enter__(self, *args, **kwargs):
        self.server = open(self.server_file, "w+", encoding="utf-8")
        return self

    def __exit__(self, *args, **kwargs):
        self.server.close()

    def write(self, s):
        print("Server:",s, end="")
        self.server.write(s)

    def log_server_transfer(self, data_length):
        self.write(f"S\t{datetime.now()}\t{data_length}\n")

    def log_server_start_next_epoch(self, epoch_number):
        self.write(f"E\t{datetime.now()}\t{epoch_number}\n")

    def log_server_message(self, message):
        self.write(f"M\t{datetime.now()}\t{message}\n")

    def log_epoch_duration(self, epoch, duration):
        self.write(f"D\t{datetime.now()}\t{epoch}\t{duration}\n")
