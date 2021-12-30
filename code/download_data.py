import os
import requests
import hashlib


def check_hash(filename, checksum):
    algorithm, value = checksum.split(':')
    if not os.path.exists(filename):
        return value, 'invalid'
    h = hashlib.new(algorithm)
    with open(filename, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            h.update(data)
    digest = h.hexdigest()
    return value, digest


# this can be found in the DOI for the Zenodo record
record_id = 5141567

# grab the urls and filenames and checksums 
r = requests.get(f"https://zenodo.org/api/records/{record_id}")
download_urls = [f['links']['self'] for f in r.json()['files']]
filenames = [(f['key'], f['checksum']) for f in r.json()['files']]

# download and verify checksums
output_dir = './forecasts'
for (fname, checksum), url in zip(filenames, download_urls):
    print(f"Downloading {fname} from Zenodo...")
    r = requests.get(url)
    full_path = os.path.join(output_dir, fname) 
    with open(full_path, 'wb') as f:
            f.write(r.content)
    value, digest = check_hash(full_path, checksum)
    if value != digest:
        print("Checksum does not match.")
