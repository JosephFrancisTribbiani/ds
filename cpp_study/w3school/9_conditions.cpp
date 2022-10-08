#include <iostream>
#include <string>
using namespace std;


int main() {
	/*
	Conditions

	C++ supports the next condition statements:
	- if
	syntax
		if (condition) {
		  // block of code to be executed if the condition is true
		}

	- else
	syntax
		if (condition) {
		  // block of code to be executed if the condition is true
		} 
		else {
		  // block of code to be executed if the condition is false
		}

	- else if
	syntax
		if (condition1) {
		  // block of code to be executed if condition1 is true
		} 
		else if (condition2) {
		  // block of code to be executed if the condition1 is false and condition2 is true
		} 
		else {
		  // block of code to be executed if the condition1 is false and condition2 is false
		}

	- switch - to specify many alternative blocks to be executed
	*/

	// example of the if statements:
	if (5 > 4) {
		cout << "5 more than 4\n";
	}

	// example of else statements
	int time;
	cout << "Enter time:\t";
	cin >> time;
	if (time < 18) {
		cout << "Good day\n";
	}
	else {
		cout << "Good evening\n";
	}

	// examle of else if statement
	cout << "Enter time:\t";
	cin >> time;
	if (time < 9) {
		cout << "Good morning\n";
	}
	else if (time < 18) {
		cout << "Good day\n";
	}
	else {
		cout << "Good evening\n";
	}

	// short version of if else statement
	// variable = (condition) ? expressionTrue : expressionFalse;
	// it's like ? equals "if" and : equals "else"
	// both ? and : required
	string results;
	cout << "Enter time:\t";
	cin >> time;
	results = (time < 18) ? "Good day\n" : "Good evening\n";
	cout << results << endl;

	return 0;
}