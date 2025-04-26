import time

def generate_filename_with_timestamp(base_name, extension):
    timestamp = time.time()
    formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
    filename = f"{base_name}_{formatted_time}.{extension}"
    return filename