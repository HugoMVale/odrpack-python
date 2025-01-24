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

# cp /mingw64/lib/pkgconfig/openblas.pc "$PKG_CONFIG_LITE/lib/pkgconfig"
# cp -r /mingw64/lib/libopenblas.* "$PKG_CONFIG_LITE/lib"
# cp -r /mingw64/include/* "$PKG_CONFIG_LITE/include"
# cp -r /mingw64/include/openblas/* "$PKG_CONFIG_LITE/include"

# Verify OpenBLAS detection
pkg-config --variable=pc_path pkg-config
pkg-config --modversion openblas
