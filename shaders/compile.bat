@echo off
setlocal enabledelayedexpansion

for %%f in (*.vert *.frag *.comp) do (
    echo Compiling %%f...
    rem Get the filename without extension using delayed expansion
    set "filename=%%~nf"
    glslc "%%f" -o "!filename!.spv"
)