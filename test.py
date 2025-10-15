import time

from tqdm import tqdm

from evotrek import evotrek


def print_log(log):
    print(log)


evotrek.init_for_sft(1)

for _ in range(2):
    progress_bar = tqdm(range(10), total=10)
    for i in progress_bar:
        time.sleep(0.5)
        elapsed = progress_bar.format_dict["elapsed"]
        elapsed_str = progress_bar.format_interval(elapsed)

    progress_bar = tqdm(range(5), total=5)
    for i in progress_bar:
        time.sleep(0.5)
        elapsed = progress_bar.format_dict["elapsed"]
        elapsed_str = progress_bar.format_interval(elapsed)
