set -xe

PROJECT_DIR="$1"
PLATFORM=$(uname -m)
echo $PLATFORM

brew install openblas

cd /System/Library/Frameworks/Accelerate.framework
ls -l
file Versions/Current/Accelerate
otool -L Versions/Current/Accelerate