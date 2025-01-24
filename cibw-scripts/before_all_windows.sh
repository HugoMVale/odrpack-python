set -xe

PROJECT_DIR="$1"

# Update pacman database and upgrade system packages
# cd c:
pwd
ls -l
PATH="/c/msys64/usr/bin:$PATH"
pacman -Sy --noconfirm
pacman -Suu --noconfirm

# Install OpenBLAS and pkg-config
pacman -S --noconfirm mingw-w64-x86_64-openblas
# pacman -S --noconfirm mingw-w64-x86_64-pkg-config

# Set PKG_CONFIG_PATH for OpenBLAS
ls -l
ls -l /c/msys64/mingw64/lib/pkgconfig/
ls -l /c/msys64/mingw64/lib/libopenblas.*
ls -l /c/msys64/mingw64/bin

# PKG_CONFIG_LITE="/c/ProgramData/Chocolatey/lib/pkgconfiglite/tools/pkg-config-lite-0.28-1"
# mkdir -p "$PKG_CONFIG_LITE/include"
# mkdir -p "$PKG_CONFIG_LITE/lib"
# mkdir -p "$PKG_CONFIG_LITE/lib/pkgconfig"

# cp /mingw64/lib/pkgconfig/openblas.pc "$PKG_CONFIG_LITE/lib/pkgconfig"
# cp -r /mingw64/lib/libopenblas.* "$PKG_CONFIG_LITE/lib"
# cp -r /mingw64/include/* "$PKG_CONFIG_LITE/include"
# cp -r /mingw64/include/openblas/* "$PKG_CONFIG_LITE/include"

# # Debugging: Verify the copied files
# echo "Contents of $PKG_CONFIG_LITE/lib/pkgconfig:"
# ls -l "$PKG_CONFIG_LITE/lib/pkgconfig"

# echo "Contents of $PKG_CONFIG_LITE/include:"
# ls -l "$PKG_CONFIG_LITE/include"

# echo "Contents of $PKG_CONFIG_LITE/lib:"
# ls -l "$PKG_CONFIG_LITE/lib"

# Convert MSYS2 path to Unix-style (in case of GitHub Actions path issues)
# export PKG_CONFIG_PATH=$(cygpath -u "C:/msys64/mingw64/lib/pkgconfig")
# echo "PKG_CONFIG_PATH is set to: $PKG_CONFIG_PATH"
# echo "PKG_CONFIG_PATH=/mingw64/lib/pkgconfig" >> $GITHUB_ENV

# Verify OpenBLAS detection
pkg-config --version
pkg-config --variable=pc_path pkg-config
pkg-config --modversion openblas
