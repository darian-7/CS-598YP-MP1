import pickle

# group_by_sum_val_list
file_path = r'/Users/darian/Desktop/UIUC/ML & Data Systems/CS598-MP1-OLA/expected_results/group_by_sum_val_list.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)
    print(data)


# # group_by_sum_key_list
# file_path = r'/Users/darian/Desktop/UIUC/ML & Data Systems/CS598-MP1-OLA/expected_results/group_by_sum_key_list.pkl'

# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
#     print(data)