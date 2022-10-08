#include <iostream>
using namespace std;


int main() {
	/*
	Arrays
	Arrays are used to store multiple values in a single variable, instead of declaring separate variables for each value.
	To declare an array, define the variable type, specify the name of the array followed by square brackets and specify
	the number of elements it should store:

		string cars[4];

	We have now declared a variable that holds an array of four strings.
	To insert values to it, we can use an array literal - place the values in a comma-separated list, inside curly braces:

		string cars[4] = {"Volvo", "BMW", "Ford", "Mazda"};

	To get elemet of an array by index, you can use:

		cars[0]

	Indexes start with 0.
	*/

	string cars[4] = {"Volvo", "Mazda", "Opel", "Ford"};
	cout << "2nd car in array is:\t" << cars[1] << endl;
	// Outputs Volvo

	/*
	You can loop through the array elements with the for loop
	*/
	for (int i = 0; i < 4; i++) {
		cout << (i + 1) << " car:\t" << cars[i] << endl;
	}
	/*
	Outputs
	1 car:  Volvo
	2 car:  Mazda
	3 car:  Opel
	4 car:  Ford
	*/

	/*
	You don't have to specify the size of the array. 
	But if you don't, it will only be as big as the elements that are inserted into it

		string cars[] = {"Volvo", "BMW", "Ford"}; // size of array is always 3

	This is completely fine. 
	However, the problem arise if you want extra space for future elements.
	Then you have to overwrite the existing values:

		string cars[] = {"Volvo", "BMW", "Ford", "Mazda", "Tesla"};

	If you specify the size however, the array will reserve the extra space:

		string cars[5] = {"Volvo", "BMW", "Ford"}; // size of array is 5, even though it's only three elements inside it

	Now you can add a fourth and fifth element without overwriting the others:
		
		cars[3] = "Mazda";
		cars[4] = "Tesla";
	*/

	string names[10] = { "Andrey", "Sergey", "Victor" };
	names[3] = "Ivan";
	for (int i = 0; i < 5; i++) {
		cout << (i + 1) << " name:\t" << names[i] << endl;
	}
	/*
	Output
	1 name: Andrey
	2 name: Sergey
	3 name: Victor
	4 name: Ivan
	5 name:
	*/

	/*
	To get the size of an array, you can use the sizeof() operator:
	It returns size of an array in bytes
	To get lenght of an array you should use:
		
		int myNumbers[5] = {10, 20, 30, 40, 50};
		int getArrayLength = sizeof(myNumbers) / sizeof(int);
		cout << getArrayLength;
	*/

	int arraySizeInBytes = sizeof(names);
	cout << "Array size in bytes:\t" << arraySizeInBytes << endl;
	int arraySize = arraySizeInBytes / sizeof(string);
	cout << "Array size:\t\t" << arraySize << endl;
	/*
	Output
	Array size in bytes:    400
	Array size:             10
	*/

	/*
	Multidimensional array
	Example:
	*/

	char letters[2][2][3] = {
		{
			{'A', 'B', 'C'},
			{'D', 'E', 'F'}
		},
		{
			{'G', 'H', 'I'},
			{'J', 'K', 'L'}
		}
	};
	cout << letters[0][1][0] << endl;  // returns D

	// Also you can change the element
	letters[0][1][0] = 'd';
	cout << letters[0][1][0] << endl;  // returns d

	// loop through multidimensional array
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 3; k++) {
				cout << letters[i][j][k] << endl;
			}
		}
	}

	return 0;
}