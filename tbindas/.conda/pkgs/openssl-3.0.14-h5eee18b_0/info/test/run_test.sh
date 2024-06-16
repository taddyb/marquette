

set -ex



touch checksum.txt
openssl sha256 checksum.txt
openssl ecparam -name prime256v1
python -c "import urllib.request; urllib.request.urlopen('https://pypi.org')"
pkg-config --print-errors --exact-version "3.0.14" openssl
exit 0
