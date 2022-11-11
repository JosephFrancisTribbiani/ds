#include <iostream>
using namespace std;


int main() {
	/*
    syntax
        switch(expression) {
          case x:
            // code block
            break;
          case y:
            // code block
            break;
          default:
            // code block
        }
	*/

    int day;
    cout << "Enter the day:\t";
    cin >> day;

    switch (day)
    {
    case 1:
        cout << "Monday\n";
        break;
    case 2:
        cout << "Tuesday\n";
        break;
    case 3:
        cout << "Wednesday\n";
        break;
    case 4:
        cout << "Thursday\n";
        break;
    case 5:
        cout << "Friday\n";
        break;
    case 6:
        cout << "Saturday\n";
        break;
    case 7:
        cout << "Sunday\n";
        break;
    // if there is no case match, then default
    default:
        cout << "Wrong day num\n";
    }

    // break is not mandatory
    // a break can save a lot of execution time because it "ignores" the execution of 
    // all the rest of the code in the switch block.

	return 0;
}