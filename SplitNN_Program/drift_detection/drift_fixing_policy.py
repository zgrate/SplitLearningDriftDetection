class ClientResponses:

    WAIT_TURNS = "wait_a_bit"
    DESTROY_CLIENT_MODEL = "destroy_model"
    REPLACE_MODEL_FROM_CLIENT = "replace_model"

class DriftFixingPolicy:

    def determine_policy(self):
        pass