import os
import shutil

import requests


LANGUAGES = ['zh', 'ja']
URL_TEMPLATE = ('https://dumps.wikimedia.org/{}wiki/latest/'
    '{}wiki-latest-pages-articles.xml.bz2')

def download_file(url):
    _, local_filename = os.path.split(url)
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print('Remote file fetched, url={}'.format(url))

for lang in LANGUAGES:
    url = URL_TEMPLATE.format(lang, lang)
    _, filename = os.path.split(url)
    if not os.path.isfile(filename):
        download_file(url)
    else:
        print('File exists; skip, filename={}'.format(filename))
