import os
import subprocess

from util.filesystem import move_file_to_archive

data_directory = "./data"


def dataset_path(data_directory, name, extension=".zip"):
    return f"{data_directory}/{name}{extension}"


if __name__ == "__main__":
    maestro_download_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    maestro_subdirectory_name = "maestro-v3.0.0"
    maestro_expanded_directory_path = dataset_path(
        data_directory, maestro_subdirectory_name, ""
    )
    maestro_zip_file_path = dataset_path(
        data_directory, maestro_subdirectory_name, ".zip"
    )

    print(maestro_subdirectory_name)
    print(maestro_expanded_directory_path)
    print(maestro_zip_file_path)

    # Create data directory if it doesn't exist.
    os.makedirs(data_directory, exist_ok=True)

    # Clear the existing copy of the dataset, if it exists.
    if os.path.exists(maestro_zip_file_path):
        rm_command = f"rm %s" % (maestro_zip_file_path)
        os.system(rm_command)
    if os.path.exists(maestro_expanded_directory_path):
        rm_command = f"rm -r %s" % (maestro_expanded_directory_path)
        os.system(rm_command)

    # Alternate commands to use depending on whether wget or curl is available.
    curl_command = f"curl -o %s %s" % (maestro_zip_file_path, maestro_download_url)
    wget_command = f"wget -O %s %s" % (maestro_zip_file_path, maestro_download_url)

    has_wget = subprocess.call("which wget", shell=True) == 0
    has_curl = subprocess.call("which curl", shell=True) == 0

    # Download Mistral MIDI dataset into the data directory.
    if has_wget:
        subprocess.call(wget_command, shell=True)
    elif has_curl:
        subprocess.call(curl_command, shell=True)
    else:
        raise Exception(
            "Neither wget nor curl is available. Please install one of these tools."
        )

    # Unzip the dataset.
    unzip_command = (
        f"unzip -d {maestro_expanded_directory_path} %s" % maestro_zip_file_path
    )
    os.system(unzip_command)

    # Archive the copy of the dataset.
    move_file_to_archive(maestro_zip_file_path)
