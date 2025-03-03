import os
import subprocess

maestro_download_url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'
maestro_subdirectory_name = 'maestro-v3.0.0'
maestro_zip_file_name = f"{maestro_subdirectory_name}.zip"
data_dir = './data'

# Create data directory if it doesn't exist.
if not os.path.exists(data_dir):
  os.makedirs(data_dir)
os.chdir(data_dir)

# Clear the existing copy of the dataset, if it exists.
if os.path.exists(maestro_subdirectory_name):
  rm_command = f"rm -r %s" % (maestro_subdirectory_name)
  os.system(rm_command)

curl_command = f"curl -o %s %s" % (maestro_zip_file_name, maestro_download_url)
wget_command = f"wget -O %s %s" % (maestro_zip_file_name, maestro_download_url)

has_wget = subprocess.call("which wget", shell=True) == 0
has_curl = subprocess.call("which curl", shell=True) == 0

# Download Mistral MIDI dataset into the data directory.
if has_wget:
  subprocess.call(wget_command, shell=True)
elif has_curl:
  subprocess.call(curl_command, shell=True)
else:
  raise Exception("Neither wget nor curl is available. Please install one of these tools.")

# Unzip the dataset.
unzip_command = f"unzip %s" % (maestro_zip_file_name)
os.system(unzip_command)

# Remove the zip file.
rm_command = f"rm %s" % (maestro_zip_file_name)
os.system(rm_command)
