@echo off
rem Launch the ferrite chat GUI (builds it first if needed).
cd /d "%~dp0"
if not exist target\release\ferrite-chat.exe (
    echo Building ferrite chat, this takes a few minutes the first time...
    cargo build --release -p ferrite-gui || exit /b 1
)
start "" target\release\ferrite-chat.exe
