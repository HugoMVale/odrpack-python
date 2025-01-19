set -xe

PROJECT_DIR="$1"

if [ -e /etc/alpine-release ]; then
    apk add --no-cache openblas-dev
else
    yum install -y libopenblas-dev
fi