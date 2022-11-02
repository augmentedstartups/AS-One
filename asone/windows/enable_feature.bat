@echo off
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
powershell.exe dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
powershell.exe dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
choco feature enable -name=exitOnRebootDetected
choco feature enable -n allowGlobalConfirmation
echo "REBOOT YOUR SYSTEM"
pause