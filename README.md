# Raspberry-Pi
---
### IoT 주차단속시스템 프로젝트에서 Raspberry-Pi(하드웨어)부분만 정리.

### 1. 작품의 개요 <br>
**1.1. 시스템 목적** <br>
IoT 주차 단속 시스템 프로젝트는 무분별한 주차 문제 및 불법 주·정차로 인한 피해를 줄이고자 진행한 프로젝트이다. <br>
프로젝트의 주요 목적으로는 소방 구역 불법 주차 위반 차량을 단속하는 것으로 중점을 뒀으며 전체적인 작동 순서는 다음과 같다. <br>
각종 센서와 카메라 모듈을 가지고 객체 탐지를 통해 차량만 감지하여 차량 단속 및 번호판 인식을 하고 <br>
인식된 번호판은 서버를 통해 데이터베이스로 단속 차량의 정보가 넘어가면 전송된 정보를 이용하여 관리자가 조치를 취하는 것으로 진행된다. <br>

**1.2. 추진 배경 및 필요성** <br>
화재 발생 시 소방용수 확보는 무엇보다 중요한데 불법 주·정차 차량으로 인하여 소방 활동 골든타임 확보에 지장을 준다. 
따라서 시민의 안전 확보를 위해 소화전 주변 불법 주·정차 행위를 근절함으로써 소화전 사용이나 구조 활동에 피해를 주지 않도록 방지해야 한다. <br>

**1.3. 기대효과** <br>
가. 작품의 기대 효과 : 신속하고 빠른 대체로 인한 세수확보 및 국민 안전 확보와 불필요한 단속인원을 줄일 수 있다. <br>
나. 참여 팀원의 교육적 기대 효과 : 프로젝트를 통해 팀원 간의 협업 능력 상승 및 문제 해결 능력을 배울 수 있다. <br>

**1.4. 활용분야** <br>
소방 관련이 아니더라도 장애인 전용 주차장이나 구급차 전용 주차장 등 주차 위반구역 같은 곳에 활용 가능하다. <br>

---
### 2. 시스템 구조 <br>
**2.1. 시스템 구성도** <br>
<img src="https://sj2565.github.io/project/images/iot.png" width="700" height="500">

**2.2. 하드웨어** <br>
2.2.1. Raspberry-Pi <br>
초음파 센서 두 개와 스피커 센서, 카메라 모듈로 전체적인 Raspberry Pi의 하드웨어를 구성한다. 각 센서들의 데이터값을 받아 서버로 전송하기 위해 사용된다. <br>
2.2.2. Echo Sensor(초음파) <br>
초음파 센서는 불법 주정차 차량의 거리를 감지하기 위해 사용되며 Raspberry Pi의 하드웨어 구성에 포함된다. <br> 센서를 사용하기 위해 Python에서 GPIO를 import하여 코드를 작성한다.
VCC전원, Trigger신호, Echo입력신호, GND 이렇게 네 개의 핀으로 구성되어 있다. <br>
2.2.3. Buzzer Sensor(스피커) <br>
스피커 센서는 초음파 센서에 거리가 감지되어도 차량을 비키지 않는 운전자들에 대한 경고 방송을 한다. <br> 
마찬가지로 센서를 사용하기 위해 GPIO를 import하며 +, - 두 개의 단자로 구성되어 있다. <br>
2.2.4. Pi Cam(카메라 모듈) <br>
카메라 모듈은 스피커 센서로 경고방송을 해도 비키지 않는 차량들에 대해 번호판을 촬영한다. <br>
Raspberry Pi에 연동되는 Pixy Cam2를 사용하며 직접 연결이 가능하다. <br>

---
### 3. 모듈 설계 <br>
**3.1. Raspberry-Pi** <br>
초음파 센서와 스피커 센서, 카메라 모듈을 사용해 각 센서들의 데이터 값을 가져오고 받은 데이터 값을 서버로 전송하기 위해 사용한다. <br> 
여러 가지 센서를 사용하여 IoT 하드웨어를 구성하고 서버와 통신을 해서 센서 정보를 자동으로 출력한다. <br>
**3.2. Echo Sensor(초음파)** <br>
while문으로 echo가 0일 때 시간을 변수 pulse_start에 저장하다가 1을 받기 직전에 그 시간이 pulse_start(start)가 된다. <br> 
while문으로 echo가 1일 때 시간을 변수 pulse_end에 저장하다가 0을 받기 직전에 그 시간이 pulse_end(stop)가 된다. <br> 
이러한 원리로 초음파가 물체에 부딪히고 돌아오는 데 걸린 시간을 구할 수 있다. <br> 
```python
pulse_duration(rtTotime) = pulse_end(stop) - pulse_start(start) 
```
**3.3. Buzzer Sensor(스피커)** <br>
반복문을 사용해서 일정 거리 미만 시 경고음 발생, 그 이상 시 경고음을 발생하지 않도록 하고 <br>
거리가 가까이 감지 될수록 경고음을 더 빨리 발생한다. <br>
**3.4. Pi Cam(카메라 모듈)** <br>
객체 탐지와 불법 주정차 차량의 번호판을 인식하기 위한 Pi Camera 모듈을 사용한다. <br>
3.4.1. 객체 탐지 <br>
객체 탐지를 통해서 차량의 정확도가 85% 이상일 때 자동 캡처를 해서 번호판 부분만 캡처되도록 한다. <br>
사용한 AI 모델은 Tensorflow 기반의 모델을 사용했다. <br>
3.4.2. 번호판 인식 <br>
카메라로 차량 번호판을 촬영 후 인식하여 번호만 추출해서 이미지 파일로 저장하고 서버로 데이터를 전달한다. <br>

