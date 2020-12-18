#include<Servo.h>

const long bud_rate = 9600; // 마스터 컴퓨터랑 똑같은 값으로 맞춰야 함
const int vel = 30; // 최소:0~최대:90
const int dir = 1; // 1(CW) or -1(CCW)                  
const int vel_com = vel*dir + 90; // 범위: 0~180
const long rot_time = 4000; // 회전을 1초간 지속

//const int space_per_snack = 2;
//const float speed_at_5v = 0.165/60;
//const long rot_time = (long)speed_at_5v*vel/90*360*space_per_snack*1000;


Servo myservo1,myservo2,myservo3,myservo4,myservo5,myservo6;
Servo myservo_array[6] = {myservo1,myservo2,myservo3,myservo4,myservo5,myservo6};
char ch;

#define echoPin 10 // attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin 9 //attach pin D3 Arduino to pin Trig of HC-SR04

// defines variables
long duration; // variable for the duration of sound wave travel
int distance; // variable for the distance measurement

const int coin = 2;
boolean insert = false;
volatile int pulse = 0;

void setup() {
  // put your setup code here, to run once:

  pinMode(trigPin, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echoPin, INPUT); // Sets the echoPin as an INPUT
  Serial.begin(9600);
  for(int i=0;i<6;i++) {
    myservo_array[i].attach(i+3);
  }
  attachInterrupt(digitalPinToInterrupt(2), coinInterrupt, RISING);
  delay(1);
}

void loop() {
  if(Serial.available()) {
    ch = Serial.read();
    if(isDigit(ch)) {
      int num = atoi(&ch);
      if(num>0 && num<7){
        myservo_array[num-1].write(vel_com);
        delay(rot_time);
        myservo_array[num-1].write(90);
      }
    }
    else if(ch == 's') {
      for(int i=0;i<6;i++) {
        myservo_array[i].write(90);
      }
      delay(10);
    }
  }
  delay(10);
  // put your main code here, to run repeatedly:
  if(insert) {
    insert = false;
    Serial.println("I");
    delay(1000);
  }
  // Clears the trigPin condition
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  // Calculating the distance
  distance = duration * 0.034 / 2; // Speed of sound wave divided by 2 (go and back)
  // Displays the distance on the Serial Monitor
  if((distance/10)>=9) distance=90;
  Serial.println(distance/10);
}

void coinInterrupt() {
  pulse++;
  insert = true;
}
