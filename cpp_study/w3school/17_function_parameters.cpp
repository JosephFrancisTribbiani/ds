#include <iostream>
using namespace std;


/*
Parameters and arguments
	Information can be passed to a function as a parameter. Parameter act as variable inside the function.
	syntax

		void functionName(type parameter1, type parameter2, type parameter3) {
		  // code to be executed
		}

Default parameters
	You can also use a default parameter value, by using the equals sign (=).

		void functionName(type parameter1, type parameter2 = default) {
		  // code to be executed
		}

	in this case parameter2 is called optional parameter
*/

void printFullName(string fname, string sname) {
	cout << fname << " " << sname << endl;
}

void printCountry(string country, string continent = "Eurasia") {
	cout << country << "\t(" << continent << ")" << endl;
}

void printAge(string name, int age) {
	cout << name << " " << age << " year\'s  old." << endl;
}

int sumValues(int x, int y) {
	return x + y;
}

void swapNums(int &x, int &y) {
	int z = x;
	x = y;
	y = z;
}

int sumOfArray(int nums[5]) {
	int s = 0;
	for (int i = 0; i < 5; i++) {
		s += nums[i];
	}
	return s;
}

int main() {
	// Parameters and arguments
	printFullName("Andrey", "Makarovskii");  // Andrey Makarovskii
	printFullName("Sergey", "Dudkin");  // Sergey Dudkin
	// when a parametr passed into a function, it's called argument
	// ("Andrei", "Makarovskii", etc. - arguments, while fname and sname - parameters)

	// Default parameters
	printCountry("USA", "North America");  // USA	(North America)
	printCountry("Russia");  // Russia	(Eurasia)

	// Mutliple parameters
	printAge("Andrey", 30);  // Andrey 30 year's  old.
	printAge("Sergey", 28);  // Sergey 28 year's  old.

	// Return values
	int val1 = 5, val2 = 6;
	cout << "x + y = " << sumValues(val1, val2) << endl;  // x + y = 11
	// also, you can assing the result to a variable
	int z = sumValues(val1, val2);
	cout << "z = " << z << endl;  // z = 11

	// Pass by reference
	// This can be useful when you need to change the value of the arguments
	int x = 5, y = 6;
	cout << "Nums before swapping:" << endl;
	cout << "x = " << x << endl;  // x = 5
	cout << "y = " << y << endl;  // y = 6
	swapNums(x, y);
	cout << "Nums after swapping:" << endl;
	cout << "x = " << x << endl;  // x = 6
	cout << "y = " << y << endl;  // y = 5

	// pass arrays as arguments into a function
	int nums[5] = {1, 4, 2, 6, 7};
	cout << sumOfArray(nums) << endl;

	return 0;
}




