# Benchmarking

## Hardware Used:
- CPU: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- GPU: 8GB (RTX2080)  

## Trackers

### DeepSort

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|DeepSort-ONNX-Yolov5s|DEEPSORT|13|3.2|
|DeepSort-Pytorch-Yolov5s|DEEPSORT|13|3.2|

### ByteTrack

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|ByteTrack-ONNX-YOLOv5s|BYTETRACK|33.7|17.4|
|ByteTrack-Pytorch-Sample-YOLOv5s|BYTETRACK|33.7|17.4|

### NorFair

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|tryolab-ONNX-YOLOv5s|NORFAIR|25.8|12|
|tryolab-Pytorch-YOLOv5s|NORFAIR|25.8|12|

### MOTPY

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|MOTPY-ONNX-YOLOv7|MOTPY|27.5|4.2|
|MOTPY-Pytorch-YOLOv7|MOTPY|32.4|3.5|

### StrongSort

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|StrongSort-ONNX-YOLOv7|STRONGSORT|7.6|3.1|
|StrongSort-Pytorch-YOLOv7|STRONGSORT|7.9|3.1|

### OCSORT

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|OCSORT-ONNX-YOLOv7|OCSORT|25.7|3.4|
|OCSORT-Pytorch-YOLOv7|OCSORT|31.4|3.2|

## Detectors
### YOLOv5

|    PyTorch                      |ONNX                         |COREML                         |
|:-------------------------------:|:-----------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV5X6_PYTORCH</td>  <td>20.8</td>  <td>3.69</td> </tr>  <tr> <td>YOLOV5S_PYTORCH</td> <td>57.25</td>  <td>25.4</td>    </tr>  <tr> <td>YOLOV5N_PYTORCH</td> <td>68</td>  <td>45</td>    </tr> <tr> <td>YOLOV5M_PYTORCH</td> <td>54</td>  <td>14</td>    </tr><tr> <td>YOLOV5L_PYTORCH</td> <td>40.06</td>  <td>8.28</td> </tr><tr> <td>YOLOV5X_PYTORCH</td> <td>28.8</td>  <td>4.32</td>    </tr><tr> <td>YOLOV5N6_PYTORCH</td> <td>63.5</td>  <td>39</td>    </tr><tr> <td>YOLOV5S6_PYTORCH</td> <td>58</td>  <td>23</td>    </tr><tr> <td>YOLOV5M6_PYTORCH</td> <td>49</td>  <td>10</td>    </tr><tr> <td>YOLOV5L6_PYTORCH </td> <td>33</td>  <td>6.5</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV5X6_ONNX</td>  <td>2.58</td>  <td>2.46</td> </tr>  <tr> <td>YOLOV5S_ONNX</td> <td>17</td>  <td>16.35</td>    </tr>  <tr> <td>YOLOV5N_ONNX</td> <td>57.25</td>  <td>35.23</td>    </tr> <tr> <td>YOLOV5M_ONNX</td> <td>45.8</td>  <td>11.17</td>    </tr><tr> <td>YOLOV5L_ONNX</td> <td>4.07</td>  <td>4.36</td> </tr><tr> <td>YOLOV5X_ONNX</td> <td>2.32</td>  <td>2.6</td>    </tr><tr> <td>YOLOV5N6_ONNX</td> <td>28.6</td>  <td>32.7</td>    </tr><tr> <td>YOLOV5S6_ONNX</td> <td>17</td>  <td>16.35</td>    </tr><tr> <td>YOLOV5M6_ONNX</td> <td>7.5</td>  <td>7.6</td>    </tr><tr> <td>YOLOV5L6_ONNX   </td> <td>3.7</td>  <td>3.98</td>    </tr> </tbody>  </table>|<table>  <thead>  <tr><th>Model Name / Model Flag</th> </tr>  </thead>  <tbody>  <tr><td>YOLOV5X6_MLMODEL</td></tr>  <tr> <td>YOLOV5S_MLMODEL</td> </tr>  <tr> <td>YOLOV5N_MLMODEL</td> </tr> <tr> <td>YOLOV5M_MLMODEL</td> </tr><tr> <td>YOLOV5L_MLMODEL</td></tr><tr> <td>YOLOV5X_MLMODEL</td></tr><tr> <td>YOLOV5N6_MLMODEL</td></tr><tr> <td>YOLOV5S6_MLMODEL</td></tr><tr> <td>YOLOV5M6_MLMODEL</td></tr><tr> <td>YOLOV5L6_MLMODEL   </td></tr> </tbody>  </table>|

### YOLOv6
|    PyTorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV6N_PYTORCH</td>  <td>65.4</td>  <td>35.32</td> </tr>  <tr> <td>YOLOV6T_PYTORCH</td> <td>63</td>  <td>15.21</td>    </tr>  <tr> <td>YOLOV6S_PYTORCH</td> <td>49.24</td>  <td>20</td>    </tr> <tr> <td>YOLOV6M_PYTORCH</td> <td>35</td>  <td>9.96</td>    </tr><tr> <td>YOLOV6L_PYTORCH</td> <td>31</td>  <td>6.2</td> </tr><tr> <td>YOLOV6L_RELU_PYTORCH</td> <td>27</td>  <td>6.3</td>    </tr><tr> <td>YOLOV6S_REPOPT_PYTORCH</td> <td>63.5</td>  <td>39</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV6N_ONNX</td>  <td>50</td>  <td>30</td> </tr>  <tr> <td>YOLOV6T_ONNX</td> <td>45.8</td>  <td>16</td>    </tr>  <tr> <td>YOLOV6S_ONNX</td> <td>41</td>  <td>13.8</td>    </tr> <tr> <td>YOLOV6M_ONNX</td> <td>25</td>  <td>6.07</td>    </tr><tr> <td>YOLOV6L_ONNNX</td> <td>17.7</td>  <td>3.32</td> </tr><tr> <td>YOLOV6L_RELU_ONNX</td> <td>19.15</td>  <td>4.36</td>    </tr><tr> <td>YOLOV6S_REPOPT_ONNX</td> <td>63.5</td>  <td>39</td>    </tr> </tbody>  </table>|

### YOLOv7
|    PyTorch                      |ONNX                         |COREML                         |
|:-------------------------------:|:-----------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV7_TINY_PYTORCH</td>  <td>53</td>  <td>19</td> </tr>  <tr> <td>YOLOV7_PYTORCH</td> <td>38</td>  <td>6.83</td>    </tr>  <tr> <td>YOLOV7_X_PYTORCH</td> <td>28</td>  <td>4.36</td>    </tr> <tr> <td>YOLOV7_W6_PYTORCH</td> <td>32.7</td>  <td>7.26</td>    </tr><tr> <td>YOLOV7_E6_PYTORCH</td> <td>15.26</td>  <td>3.07</td> </tr><tr> <td>YOLOV7_D6_PYTORCH</td> <td>21</td>  <td>3.78</td>    </tr><tr> <td>YOLOV7_E6E_PYTORCH</td> <td>24</td>  <td>3.36</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV7_TINY_ONNX</td>  <td>41.6</td>  <td>22</td> </tr>  <tr> <td>YOLOV7_ONNX</td> <td>26</td>  <td>3.78</td>    </tr>  <tr> <td>YOLOV7_X_ONNX</td> <td>19.08</td>  <td>2.35</td>    </tr> <tr> <td>YOLOV7_W6_ONNX</td> <td>28.6</td>  <td>5.2</td>    </tr><tr> <td>YOLOV7_E6_ONNX</td> <td>14.3</td>  <td>2.97</td> </tr><tr> <td>YOLOV7_D6_ONNX</td> <td>18.32</td>  <td>2.58</td>    </tr><tr> <td>YOLOV7_E6E_ONNX</td> <td>15.26</td>  <td>2.09</td>    </tr> </tbody>  </table>|<table>  <thead>  <tr><th>Model Name / Model Flag</th> </tr>  </thead>  <tbody>  <tr><td>YOLOV7_TINY_MLMODEL</td></tr>  <tr> <td>YOLOV7_MLMODEL</td> </tr>  <tr> <td>YOLOV7_X_MLMODEL</td> </tr> <tr> <td>YOLOV7_W6_MLMODEL</td> </tr><tr> <td>YOLOV7_E6_MLMODEL</td></tr><tr> <td>YOLOV7_D6_MLMODEL</td></tr><tr> <td>YOLOV7_E6E_MLMODEL</td></tr></tbody>  </table>|

### YOLOR
|    Pytorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOR_CSP_X_PYTORCH</td>  <td>28.6</td>  <td>1.83</td> </tr>  <tr> <td>YOLOR_CSP_X_STAR_PYTORCH</td> <td>30</td>  <td>1.76</td>    </tr>  <tr> <td>YOLOR_CSP_STAR_PYTORCH</td> <td>38.1</td>  <td>2.86</td>    </tr> <tr> <td>YOLOR_CSP_PYTORCH</td> <td>38</td>  <td>2.77</td>    </tr><tr> <td>YOLOR_P6_PYTORCH</td> <td>20</td>  <td>1.57</td> </tr></tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOR_CSP_X_ONNX</td>  <td>15.7</td>  <td>2.53</td> </tr>  <tr> <td>YOLOR_CSP_X_STAR_ONNX</td> <td>15.79</td>  <td>2.05</td>    </tr>  <tr> <td>YOLOR_CSP_STAR_ONNX</td> <td>18.32</td>  <td>3.34</td>    </tr> <tr> <td>YOLOR_CSP_ONNX</td> <td>15.7</td>  <td>2.53</td>    </tr><tr> <td>YOLOR_P6_ONNX</td> <td>25.4</td>  <td>5.58</td> </tr></tbody>  </table>|

### YOLOX
|    Pytorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOX_L_PYTORCH</td>  <td>2.58</td>  <td>2.31</td> </tr>  <tr> <td>YOLOX_NANO_PYTORCH</td> <td>35</td>  <td>32</td>    </tr>  <tr> <td>YOLOX_TINY_PYTORCH</td> <td>25.4</td>  <td>25.4</td>    </tr> <tr> <td>YOLOX_DARKNET_PYTORCH</td> <td>2</td>  <td>1.94</td>    </tr><tr> <td>YOLOX_S_PYTORCH</td> <td>9.54</td>  <td>9.7</td> </tr><tr> <td>YOLOX_M_PYTORCH</td> <td>4.4</td>  <td>4.36</td>    </tr><tr> <td>YOLOX_X_PYTORCH</td> <td>15.64</td>  <td>1.39</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOX_L_ONNX</td>  <td>22.9</td>  <td>3.07</td> </tr>  <tr> <td>YOLOX_NANO_ONNX</td> <td>59</td>  <td>54</td>    </tr>  <tr> <td>YOLOX_TINY_ONNX</td> <td>60</td>  <td>35</td>    </tr> <tr> <td>YOLOX_DARKNET_ONNX</td> <td>24</td>  <td>3.36</td>    </tr><tr> <td>YOLOX_S_ONNX</td> <td>45</td>  <td>13.8</td> </tr><tr> <td>YOLOX_M_ONNX</td> <td>32</td>  <td>6.54</td>    </tr><tr> <td>YOLOX_X_ONNX</td> <td>15.79</td>  <td>2.03</td>    </tr> </tbody>  </table>|

### YOLOv8
|    Pytorch                      |ONNX                         |COREML                         |
|:-------------------------------:|:-----------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV8N_PYTORCH</td>  <td>26.7</td>  <td>17.0</td> </tr>  <tr> <td>YOLOV8S_PYTORCH</td> <td>26.4</td>  <td>12.3</td>    </tr>  <tr> <td>YOLOV8M_PYTORCH</td> <td>25.1</td>  <td>6.8</td>    </tr> <tr> <td>YOLOV8L_PYTORCH</td> <td>23.6</td>  <td>4.0</td>    </tr><tr> <td>YOLOV8X_PYTORCH</td> <td>20.7</td>  <td>2.8</td> </tr><tr></tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV8N_ONNX</td>  <td>25.1</td>  <td>10.5</td> </tr>  <tr> <td>YOLOV8S_ONNX</td> <td>24.5</td>  <td>7.5</td>    </tr>  <tr> <td>YOLOV8M_ONNX</td> <td>22.9</td>  <td>4.7</td>    </tr> <tr> <td>YOLOV8l_ONNX</td> <td>20.4</td>  <td>2.9</td>    </tr><tr> <td>YOLOV8X_ONNX</td> <td>19.0</td>  <td>2.0</td> </tr> </tbody>  </table>|<table>  <thead>  <tr><th>Model Name / Model Flag</th> </tr>  </thead>  <tbody>  <tr><td>YOLOV8N_MLMODEL</td>  </tr>  <tr> <td>YOLOV8S_MLMODEL</td> </tr>  <tr> <td>YOLOV8M_MLMODEL</td>  </tr> <tr> <td>YOLOV8L_MLMODEL</td></tr><tr> <td>YOLOV8X_MLMODEL</td></td> </tr> </tbody>  </table>|

## OCR

| Model           |  Model Flag 
|---------------- |-----------| 
|Craft|CRAFT|
|dbnet18|DBNET18


Return to [Installation Page](../../../README.md) 
