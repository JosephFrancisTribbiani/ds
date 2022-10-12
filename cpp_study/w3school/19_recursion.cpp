#include <iostream>
using namespace std;


/*
Recursion is the technique of making a function call itself. 
This technique provides a way to break complicated problems down into simple problems which are easier to solve.
*/

int factorial(int k) {
	if (k > 1) {
		return k * factorial(k - 1);
	}
	else {
		return 1;
	}
}


int main() {
	// Recursion Example
	// factorial calculation
	cout << "0! = " << factorial(0) << endl;  // 1
	cout << "1! = " << factorial(1) << endl;  // 1
	cout << "2! = " << factorial(2) << endl;  // 2
	cout << "3! = " << factorial(3) << endl;  // 6
	cout << "4! = " << factorial(4) << endl;  // 24
	cout << "5! = " << factorial(5) << endl;  // 120

	return 0;
}