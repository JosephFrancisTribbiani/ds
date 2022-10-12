#include <iostream>
using namespace std;


/*
Functions
create a function
	syntax:

		void functionName() {  // declaration of the function
		  // code to be executed - function definition
		}

	functionName - name of the function
	void means that function does not return anything

function declaration and definition
	if you declare a function after main(), you will get an error
	for example

		int main() {
			myFunction();
			return 0;
		}

		void myFunction() {  // declaration
			// definition
		}

	but you can declare a function before main and define the function after main
	then the code will be executed without any error
	example

		void myFunction();  // declaration

		int main() {
			myFunction();
			return 0;
		}

		void myFunction() {
			// definition
		}
*/

void greetings() {  // greetings function declaration
	cout << "Hello, world" << endl;  // greetings function definition
}

void goodbye();  // goodbye function declaration

int main() {
	greetings();  // outputs "Hello, world"
	goodbye();  // outputs "Goodbye, world"
	return 0;
}

void goodbye() {
	cout << "Goodbye, world" << endl;  // goodbye function definition
}
