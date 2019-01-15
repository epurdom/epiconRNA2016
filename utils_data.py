import requests
import os
from requests.auth import HTTPBasicAuth


def fetch_data(user="", password=""):
    base_url = "https://www.stat.berkeley.edu/~epicon/publications/rnaseq/"
    leaf_data = "data/RNASeq/leaf_log.tsv"
    _fetch_single_data(base_url + leaf_data, leaf_data, user=user,
                       password=password)

    leaf_meta = "data/RNASeq/leaf_meta.tsv"
    _fetch_single_data(base_url + leaf_meta, leaf_meta, user=user,
                       password=password)

    root_data = "data/RNASeq/root_log.tsv"
    _fetch_single_data(base_url + root_data, root_data, user=user,
                       password=password)

    root_meta = "data/RNASeq/root_meta.tsv"
    _fetch_single_data(base_url + root_meta, root_meta, user=user,
                       password=password)


def _fetch_single_data(url, outname, user="", password="", verbose=True):
    if verbose:
        print("Fetching data file: %s" % url)
    r = requests.get(url,
                     auth=HTTPBasicAuth(user, password),
                     allow_redirects=True)
    if r.status_code == 401:
        raise ValueError(
            "username or password provided is wrong (%s, %s)" %
            (user, password))
    try:
        os.makedirs(os.path.dirname(outname))
    except OSError:
        pass

    with open(outname, "wb") as f:
        f.write(r.content)
