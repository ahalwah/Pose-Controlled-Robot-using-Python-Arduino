#include <SoftwareSerial.h>
//UART TO HM10 Module
const int bluRX_ardTXpin = 3;
const int bluTX_ardRXpin = 2;
SoftwareSerial bluetooth(bluTX_ardRXpin, bluRX_ardTXpin);
String x;

void setup() {
 bluetooth.begin(9600);
 Serial.begin(9600);
 Serial.setTimeout(1);
 pinMode(led,OUTPUT);
 digitalWrite(led, LOW);
}
void loop() {
 while (!Serial.available());
 char charBuffer[20];//most we would ever see
 int numberOfBytesReceived = Serial.readBytesUntil('\n', charBuffer, 19);
 charBuffer[numberOfBytesReceived] = NULL;
 int index = numberOfBytesReceived-1;
 //x = Serial.readString();
 //Serial.print(charBuffer[index]);
 bluetooth.write(charBuffer[index]);
}
