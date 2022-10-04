#include <iostream> // библиотека потокового ввода/вывода данных
#include <string> // для использования строк необходимо добавить дополнительную библиотеку
using namespace std;

/*
В C++ несколько типов данных:
int - целочисленное число, 2 или 4 байта
float - число с плавающей точкой, 4 байта
double - число с плавающей точкой, 8 байт
char - однин символ, например 'd' - 1 байт (use single quotes)
string - строка, например "Hello world" (use double quotes)
bool - булевый тип данных true или false - 1 байт
*/

int main() {
	// целочисленное число
	int myInt = 5;
	cout << "int value: " << myInt << endl;

	// с плавающей точкой float
	double myFloat_1 = 0.123456789, myFloat_2 = 1e2, myFloat_3 = 1e-2; // or you can write 1E2 and 1E-2
	cout << "first float value: " << myFloat_1 << endl;
	cout << "second float value: " << myFloat_2 << endl;
	cout << "third float value: " << myFloat_3 << endl;

	// с плавающей точкой double
	double myDouble = 0.123456789;
	cout << "double value: " << myDouble << endl;

	// char
	char myChar = 'd';
	cout << "char value: " << myChar << endl;
	// or ASCII values as char value
	// list of ASCII values you can find in the ASCII table
	char myASCII = 72;
	cout << "ASCII as char value: " << myASCII << endl;

	// string
	string myString = "Hello world";
	cout << "string value: " << myString << endl;

	// bool
	bool myBool_1 = true, myBool_2 = "Hi", myBool_3 = 0;
	cout << "first boolean value: " << myBool_1 << endl;
	cout << "second boolean value: " << myBool_2 << endl;
	cout << "third boolean value: " << myBool_3 << endl;
	return 0;
}