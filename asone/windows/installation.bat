@echo on
choco install wsl2 --params "/Version:2 /Retry:true"
wsl --set-default-version 2
choco install docker-desktop
choco install vcxsrv
for /F "tokens=14" %i in ('"ipconfig | findstr IPv4 | findstr /i "192""') do setx DISPLAY %i:0.0
echo "REBOOT YOUR SYSTEM"
Pause
