from urllib import request, parse
import json
import os

lookup = {
    1:"bats",               # Bat
    11:"eagle",             # Eagle
    6:"cicadidae",          # Cicada
    12:"psittaciformes",    # Parrot
    16:"archilochus",       # Hummingbird
    19:"orcinus+orca",      # Killer Whale
    20:"macaca",            # Macaque
    10:"dolphin",           # Dolphin
}

for k in lookup:

    name = lookup[k]

    # Make folder for class
    try:
        class_dir = os.path.join("data", str(k))
        os.mkdir(os.path.join("data", str(k)))
    except FileExistsError:
        print(f"{name} already exists")
        continue

    print(f"Downloading {name} sounds...")
    tax_url = f"https://taxonomy.api.macaulaylibrary.org/v1/taxonomy?key=PUB4334626458&q={name}"
    with request.urlopen(tax_url) as r:
        resp = json.loads(r.read())
        codes = [entry["code"] for entry in resp]
    
    for taxon_code in codes:
        old_cursor = None
        cursor = ""

        # Get urls
        all_ids = []
        while old_cursor != cursor:
            api_url = f"https://search.macaulaylibrary.org/catalog.json?taxonCode={taxon_code}&count=100&mediaType=a&initialCursorMark={parse.quote(cursor)}"
            with request.urlopen(api_url) as r:
                resp = json.loads(r.read())
                old_cursor = cursor
                cursor = resp['results']["nextCursorMark"]
                for entry in resp["results"]["content"]:
                    rating = 0 if entry["rating"] == None else float(entry["rating"])
                    if rating >= 4.0:
                        all_ids.append( (entry["assetId"], entry["mediaUrl"]) )

        # Download
        for a_id, asset_url in all_ids:
            print(asset_url)
            request.urlretrieve(asset_url, os.path.join(class_dir, f"{a_id}.mp3"))

print("Done!")
