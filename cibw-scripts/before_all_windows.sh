set -xe

# Verify compiler locations
which gcc
which gfortran

# Update pacman database and upgrade system packages
PATH="/c/msys64/usr/bin:$PATH"
pacman -Sy --noconfirm
pacman -Suu --noconfirm

# Install OpenBLAS
pacman -S --noconfirm mingw-w64-x86_64-openblas

ls -l
ls -l /c/msys64/mingw64/lib/pkgconfig/
ls -l /c/msys64/mingw64/lib/libopenblas.*
ls -l /c/msys64/mingw64/bin

cp /c/msys64/mingw64/bin/libopenblas.dll /c/mingw64/bin/

# Verify OpenBLAS detection
pkg-config --variable=pc_path pkg-config
pkg-config --modversion openblas
