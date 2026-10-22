$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$LlvmVersion = '22.1.8'
$LlvmBin = 'C:\Program Files\LLVM\bin'
$LlvmInstaller = "$env:RUNNER_TEMP\LLVM-22.1.8-woa64.exe"
$LlvmUrl = "https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/LLVM-22.1.8-woa64.exe"
Invoke-WebRequest $LlvmUrl -UseBasicParsing -OutFile $LlvmInstaller
$ExpectedHash = "76f44ef1ba6eeb5a65904e9500f042f588fade49952778ce48f0374daa934396"
$FileHash = (Get-FileHash -Path $LlvmInstaller -Algorithm SHA256).Hash
if ($FileHash -ne $ExpectedHash) {
    Remove-Item $LlvmInstaller -Force -ErrorAction SilentlyContinue
    throw "LLVM SHA256 verification failed. Installation aborted."
}
Start-Process -FilePath $LlvmInstaller -ArgumentList '/S' -Wait
$env:PATH = "$LlvmBin;$env:PATH"
$LlvmBin | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
$LlvmRuntime = "$LlvmBin\..\lib\clang\22.1.8\lib\windows"
$FlangRuntime = "$LlvmBin\..\lib\clang\22\lib\aarch64-pc-windows-msvc"
$env:LIB = "$FlangRuntime;$LlvmRuntime;$env:LIB"
"LIB=$env:LIB" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
@(
    'CC=clang-cl'
    'CXX=clang-cl'
    'FC=flang-new'
    'AR=llvm-ar'
) | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
