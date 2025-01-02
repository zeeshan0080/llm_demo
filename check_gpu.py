import torch
import time

def format_seconds_to_minutes(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} minutes and {remaining_seconds} seconds"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
print(device)

start_time = time.time()
time.sleep(10)
end_time = time.time()
elapsed_time = (end_time - start_time)
formatted_time = format_seconds_to_minutes(elapsed_time)
print(formatted_time)