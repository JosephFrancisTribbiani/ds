#include <iostream>
using namespace std;


int main() {
	/*
	while loop
	syntax
		while (condition) {
		  // code block to be executed
		}

	do/while loop
	variant of the while loop
	difference is than it will execute the code block once, before checking if the condition is true, 
	then it will repeat the loop as long as the condition is true 
	syntax
		do {
		  // code block to be executed
		}
		while (condition);

	for loop
	syntax
		for (statement 1; statement 2; statement 3) {
		  // code block to be executed
		}
	statement 1 is executed (one time) before the execution of the code block.
	statement 2 defines the condition for executing the code block.
	statement 3 is executed (every time) after the code block has been executed.
	*/

	// while loop example
	int i = 0;
	cout << "Welcome to while loop\n";
	while (i < 5) {
		cout << "i = " << i << endl;
		i++;
	}

	// do/while loop example
	i = 0;
	cout << "Welcome to do//while loop\n";
	do {
		cout << "i = " << i << endl;
		i++;
	} while (i < 1);

	// for loop example
	cout << "Welcome to for loop\n";
	for (i = 0; i < 5; i++) {
		cout << "i = " << i << endl;
	}

	// break
	// if you want to interupt the loop, you can use break
	// example in while loop
	cout << "Break example\n";
	i = 0;
	while (true) {  // infinity loop
		cout << "i = " << i << endl;
		i++;
		if (i == 3) {
			cout << "i euals 3 - breaking the loop\n";
			break;
		}
	}

	// continue
	// if you want skip loop, for example, 
	// you can use continue
	// example in for loop
	cout << "Continue example\n";
	for (i = 0; i < 5; i++) {
		if (i == 3) {
			cout << "Skipping i == 3 because of continue\n";
			continue;
		}
		cout << "i = " << i << endl;
	}
	return 0;
}