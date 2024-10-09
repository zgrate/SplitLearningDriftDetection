# import openmi
#
# # Compute private set intersection
# client_items = dataloader.dataloader1.dataset.get_ids()
# server_items = dataloader.dataloader2.dataset.get_ids()
#
# client = Client(client_items)
# server = Server(server_items)
#
# setup, response = server.process_request(client.request, len(client_items))
# intersection = client.compute_intersection(setup, response)
#
# # Order data
# dataloader.drop_non_intersecting(intersection)
# dataloader.sort_by_ids()