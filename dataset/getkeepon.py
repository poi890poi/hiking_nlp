import os
import shutil
import hashlib
import lxml
import lxml.html
import re
import json
import time
import random
import traceback

import requests


URL_TEMPLATE_INDEX = 'https://www.keepon.com.tw/forum-{}-{}.html'
URL_TEMPLATE_ARTICLE = 'https://www.keepon.com.tw'
FORUMS = [
    (1, '登山行程紀錄'),
    (2, '路況回報'),
    (3, '藍天圖集'),
    (4, '祥馬圖集'),
    (5, '流浪山裡的故事'),
    (6, '台灣行腳'),
    (9, '台灣登山論壇'),
]
RE_LAST_PAGE = re.compile('gotoPage\((\d+)\)')

def download_file(url, description, folder, exinfo=None):
    fetched_from_remote = False
    _, ext = os.path.splitext(url)
    result = hashlib.md5(url.encode())
    md5 = result.hexdigest()
    local_filename = '{}{}'.format(md5, ext)
    directory = os.path.join('./keepon', folder, md5[:2], md5[2:4])
    os.makedirs(directory, exist_ok=True)
    local_filename = os.path.join(directory, local_filename)
    if not os.path.isfile(local_filename):
        try:
            with requests.get(url) as r:
                fetched_from_remote = True
                if r.status_code == 200:
                    with open(local_filename, 'wb') as f:
                        f.write(r.content)
                        print('Remote file fetched, url={}, filename={}'.format(url, local_filename))
                else:
                    print('HTTP error, code={}, url={}'.format(r.status_code, url))
        except:
            traceback.print_exc()
    else:
        print('File exists; skip, filename={}'.format(local_filename))

    # Save meta info
    try:
        meta_filename = '{}{}'.format(md5, '.json')
        meta_filename = os.path.join(directory, meta_filename)
        force_update_meta = True
        if force_update_meta or not os.path.isfile(meta_filename):
            with open(meta_filename, 'w') as f:
                meta = {
                    'url': url,
                    'timestamp': time.time(),
                    'description': description,
                }
                if exinfo is not None:
                    meta.update(exinfo)
                json.dump(meta, f)
    except:
        traceback.print_exc()

    return local_filename, fetched_from_remote

page_max = -1
page = 1
for f in FORUMS[:1]:
    forum_id, forum_description = f
    while page_max == -1 or page <= page_max:
        # Fetch and parse index page
        url = URL_TEMPLATE_INDEX.format(forum_id, page)
        try:
            local_filename, fetched_from_remote = download_file(url, forum_description, 'index')
            tree = lxml.html.parse(local_filename)
            if page_max == -1:
                for c in tree.xpath('//ul/comment()'):
                    m = RE_LAST_PAGE.findall(c.text)
                    if m:
                        m = int(m[0])
                        if m > page_max:
                            page_max = m
                            print('Forum page max found, page={}'.format(page_max))
            articles = tree.xpath('//h4/a')
        except:
            traceback.print_exc()
            continue

        # Fetch all articles in this index page
        for p in articles:
            try:
                caption = p.get('title')
                href = p.get('href').strip('/')
                if 'thread' in href:
                    url = os.path.join(URL_TEMPLATE_ARTICLE, href)
                    print(page, caption, url)
                    article_filename, fetched_from_remote = download_file(
                        url, caption, 'articles', exinfo={'forum_id': forum_id})
                    if fetched_from_remote:
                        time.sleep(random.randint(60, 120))
            except:
                traceback.print_exc()

        # Next page...
        page += 1
