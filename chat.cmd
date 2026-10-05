@echo off
rem Launch ferrite chat (the Tauri desktop app), building it first if needed.
rem   chat.cmd           start (build only if the exe is missing)
rem   chat.cmd rebuild   rebuild, then start
cd /d "%~dp0"
set "WINLIBS=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
if /i "%~1"=="rebuild" goto build
if exist target\release\ferrite-chat.exe goto run
:build
echo Building ferrite chat (a few minutes the first time)...
rem WinLibs' 64-bit windres/dlltool must come before the old 32-bit C:\MinGW on PATH
set "PATH=%WINLIBS%;%PATH%"
cargo build --release -p ferrite-app || exit /b 1
:run
start "" target\release\ferrite-chat.exe
