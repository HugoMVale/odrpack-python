set -xe

WHEEL=$1
DEST_DIR=$2

python -m delvewheel show "$WHEEL" 
python -m delvewheel repair -w "$DEST_DIR" "$WHEEL" --add-path "C:\msys64\mingw64\bin\libopenblas.dll"