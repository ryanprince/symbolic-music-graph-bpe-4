import os
import shutil
import datetime

archive_directory = "./archive"


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).__str__()


def timestamp_file_name(file_name):
    return f"{file_name} {timestamp()}"


def archive_file_path(file_path):
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1]
    archive_path = os.path.join(
        archive_directory, timestamp_file_name(file_name) + file_extension
    )
    return archive_path


def _archive(file_path, mode):
    os.makedirs(archive_directory, exist_ok=True)
    mode(file_path, archive_file_path(file_path))


def move_file_to_archive(file_path):
    _archive(file_path, shutil.move)


def copy_file_to_archive(file_path):
    _archive(file_path, shutil.copyfile)
