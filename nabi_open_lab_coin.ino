#define echoPin 4 // attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin 3 //attach pin D3 Arduino to pin Trig of HC-SR04

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
  attachInterrupt(digitalPinToInterrupt(2), coinInterrupt, RISING);
  delay(1000);
}

void loop() {
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
  Serial.println(distance/10);
}

void coinInterrupt() {
  pulse++;
  insert = true;
}
