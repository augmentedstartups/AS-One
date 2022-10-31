# Setting ASOne on Docker

1. Clone the repo

```
git clone https://github.com/axcelerateai/asone.git
cd asone
```

2. If using windows, Run this command in command prompt.
```
set PWD=%cd%
```
2. Run docker coompose command.

```
# To test on Linux with GPU 
docker compose run linux-gpu

# To test on Windows with GPU 
docker compose run windows-gpu
```

```
# To test on Linux with CPU 
docker compose run linux

# To test on Windows with CPU 
docker compose run windows
```

3. In docker terminal.

```
# if using gpu
python main.py [VIDEO_PATH]

# if using cpu
python main.py [VIDEO_PATH] --cpu
```

Return to [main page](../../README.md)
