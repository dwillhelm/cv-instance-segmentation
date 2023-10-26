#!/bin/bash

# download the livecell dataset.

#
curl -H "GET /?list-type=2 HTTP/1.1" \
     -H "Host: livecell-dataset.s3.eu-central-1.amazonaws.com" \
     -H "Date: 20161025T124500Z" \
     -H "Content-Type: text/plain" http://livecell-dataset.s3.eu-central-1.amazonaws.com/ > files.xml

#
grep -oPm1 "(?<=<Key>)[^<]+" files.xml | sed -e 's/^/http:\/\/livecell-dataset.s3.eu-central-1.amazonaws.com\//' > urls.txt

# wget URLs
while read -r line; do curl -O $line; done < urls.txt