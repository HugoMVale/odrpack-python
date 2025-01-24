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

ls -l /c/msys64/mingw64/lib/pkgconfig/
ls -l /c/msys64/mingw64/lib/libopenblas.*
ls -l /c/msys64/mingw64/bin

# Copy OpenBLAS files to mingw64 (where the active compiler toolchain is)
mkdir -p /c/mingw64/lib/pkgconfig
cp /c/msys64/mingw64/lib/pkgconfig/openblas.pc /c/mingw64/lib/pkgconfig/
cp /c/msys64/mingw64/bin/libopenblas.dll /c/mingw64/bin/
cp /c/msys64/mingw64/lib/libopenblas.*   /c/mingw64/lib/
cp /c/msys64/mingw64/include/openblas    /c/mingw64/include

# Verify OpenBLAS detection
pkg-config --modversion openblas
