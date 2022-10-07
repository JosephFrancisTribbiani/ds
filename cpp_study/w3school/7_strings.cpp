#include <iostream>
#include <string>  // для полноценной работы со строками необходимо добавить эту библиотеку
using namespace std;

int main() {
	// создание первой строки
	string greeting = "Hello";
	cout << greeting << endl;

	// конкатенация строк
	string firstName = "John";
	string lastName = "Doil";
	cout << firstName + " " + lastName << endl;

	// вообщето, строки это экземпляры классов, и они имеют свои методы
	string fullName = firstName.append(" ");
	fullName.append(lastName);
	cout << fullName << endl;

	// число и строку конкатенировать нельзя
	// т.е. "20" + 10 выдаст ошибку

	// длину строки можно определить с помощью методов length() или size()
	string txt = "veryyyyloooongstring";
	cout << "Using method length:\t" << txt.length() << endl;
	cout << "Using method size:\t" << txt.size() << endl;

	// to access string character by index use square quotes
	cout << "Third char of very long string:\t" << txt[2] << endl;
	// the first index begins from 0

	// also you can change the single string char
	txt[2] = 'A';
	cout << "Third char of very long string:\t" << txt[2] << endl;

	// special characters:
	// \n - new line (example: "Hello\n")
	// \t - tabular (example: "Answer:\t5")
	// escape character for ', ", \ (example: "It\'s me")

	// to get string entered by user you can use cin
	// but space character (whitespaces, tabs etc.) terminates user input
	// for example if you type "John Snow" you will get "Jonh" only
	// 
	// to parse entire string use getline() function (see example below)
	cout << "Enter your full name:" << endl;
	getline(cin, fullName);
	cout << "Your name is:\t" << fullName << endl;
	// results:
	// Enter your full name:
	// John Snow
	// Your name is:	John Snow
	return 0;
}