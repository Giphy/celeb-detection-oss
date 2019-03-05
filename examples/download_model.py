import logging
import os
import tarfile
import urllib.request


s3_bucker_dir = 'https://s3.amazonaws.com/giphy-public/models/celeb-detection/'
archive_name = 'resources.tar.gz'


def download():
    archive = urllib.request.urlopen(s3_bucker_dir + archive_name)
    with open(archive_name, 'wb') as f:
        f.write(archive.read())
    with tarfile.open(archive_name, "r:gz") as tar:
        tar.extractall()
    os.remove(archive_name)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    if not os.path.isdir(archive_name.split('.tar.gz')[0]):
        logging.info('Starting to download archive from S3')
        download()
        logging.info('Done')
    else:
        logging.info('Resources directory already exists')
