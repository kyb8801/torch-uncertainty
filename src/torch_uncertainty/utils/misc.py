import csv
from pathlib import Path


def csv_writer(path: Path, metrics_dict: dict) -> None:
    """Write a dictionary of metric values to a csv file.

    Args:
        path: Path to the csv file.
        metrics_dict: Dictionary to write.
    """
    if path.is_file():  # Check that the file already exists
        append_mode = True
        rw_mode = "a"
    else:
        append_mode = False
        rw_mode = "w"
    # Write metrics_dict
    with path.open(rw_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # Do not write header in append mode
        if append_mode is False:
            writer.writerow(metrics_dict.keys())
        writer.writerow([f"{elem:.4f}" for elem in metrics_dict.values()])
