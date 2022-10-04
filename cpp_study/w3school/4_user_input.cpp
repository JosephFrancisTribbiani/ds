#include <iostream>
using namespace std;

// для ввода данных с клавиатуры пользователем используется команда cin

int main() {
	int x, y;
	cout << "Enter the first digit: ";
	cin >> x;
	cout << "Enter the second digit: ";
	cin >> y;
	cout << "The sum is: " << x + y;
	return 0;
}