"""FedCondGraphRAG package."""


def load_task(args, client_id, data, data_dir, device):
    from fedcond_grag.server.stage_c_aggregate.task import CondensationQATask
    return CondensationQATask(args, client_id, data, data_dir, device)


def load_client(args, client_id, data, data_dir, message_pool, device):
    from fedcond_grag.client.client import FedCondQAClient
    return FedCondQAClient(args, client_id, data, data_dir, message_pool, device)


def load_server(args, global_data, data_dir, message_pool, device):
    from fedcond_grag.server.server import FedCondQAServer
    return FedCondQAServer(args, global_data, data_dir, message_pool, device)
