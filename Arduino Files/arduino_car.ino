#include <SoftwareSerial.h>
//UART TO HM10 Module
const int bluRX_ardTXpin = 2;
const int bluTX_ardRXpin = 3;
SoftwareSerial bluetooth(bluTX_ardRXpin, bluRX_ardTXpin);
//Pins 
const int motorOne_InputTwo=4;
const int motorOne_InputOne=5;
const int motorOne_Enable=9;
const int motorTwo_Enable=10;
const int motorTwo_InputOne=6;
const int motorTwo_InputTwo=7;

int led=13;
void setup() {
  // put your setup code here, to run once:
  for(int i=4; i<=10; i++){
    if(i==8)
      continue;
  pinMode(i,OUTPUT);
  }
  //motor speeds
  digitalWrite(motorOne_Enable, HIGH);
  digitalWrite(motorTwo_Enable, HIGH);
  pinMode(led,OUTPUT);
  digitalWrite(led,LOW);
  bluetooth.begin(38400);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  char charBuffer[20];//most we would ever see
  if(bluetooth.available()>0){
    //indexing starts at 0 so if we read in 'cool' we have 4 bytes 
    //but we are going from 0 to 3 therefore at index 4 we set to null
    //then read only from 0 to 3 to retrieve our input of 'cool'
    int numberOfBytesReceived = bluetooth.readBytesUntil('\n', charBuffer, 19);
    charBuffer[numberOfBytesReceived] = NULL;
    int index = numberOfBytesReceived-1;
    Serial.println(charBuffer[index]);
    if(charBuffer[index]=='S'){
      digitalWrite(motorTwo_InputOne,LOW);
      digitalWrite(motorTwo_InputTwo,LOW);
      digitalWrite(motorOne_InputOne,LOW);
      digitalWrite(motorOne_InputTwo,LOW);
    }
    if(charBuffer[index]=='R'){
      digitalWrite(motorTwo_InputOne,HIGH);
      digitalWrite(motorTwo_InputTwo,LOW);
      digitalWrite(motorOne_InputOne,LOW);
      digitalWrite(motorOne_InputTwo,HIGH);
    }
    if(charBuffer[index]=='L'){
      digitalWrite(motorTwo_InputOne,LOW);
      digitalWrite(motorTwo_InputTwo,HIGH);
      digitalWrite(motorOne_InputOne,HIGH);
      digitalWrite(motorOne_InputTwo,LOW);
    }
    if(charBuffer[index]=='F'){
      digitalWrite(motorTwo_InputOne,HIGH);
      digitalWrite(motorTwo_InputTwo,LOW);
      digitalWrite(motorOne_InputOne,HIGH);
      digitalWrite(motorOne_InputTwo,LOW);
    }
    if(charBuffer[index]=='B'){
      digitalWrite(motorTwo_InputOne,LOW);
      digitalWrite(motorTwo_InputTwo,HIGH);
      digitalWrite(motorOne_InputOne,LOW);
      digitalWrite(motorOne_InputTwo,HIGH);
    }
  }
}
