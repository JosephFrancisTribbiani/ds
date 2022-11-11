#include <cmath>
#include <iostream>
using namespace std;


int main() {
	// functions (you should add cmath library to use them):
	// abs(x)
	// acos(x)
	// asin(x)
	// atan(x)
	// cbrt(x) - cube root
	// ceil(x) - round up to nearest integer
	// cos(x)
	// cosh(x) - hyperbolic cos
	// exp(x)
	// expm1(x) - returns exp(x) - 1
	// fabs(x) - returns absolute value of a floating x value
	//			 abs dealing with both - int and float
	//			 fabs in pre 11 version of C++ dealing with float values only
	// fdim(x, y) - returns the positive difference between x and y
	//				if x > y, returns x - y
	//				else returns 0
	// floor(x) - round down to nearest int
	// hypot(x, y) - sqrt(x2 +y2) without intermediate overflow or underflow
	// fma(x, y, z) - x*y+z without losing precision
	// fmax(x, y) - highest value of a floating x and y
	// fmin(x, y) - lowest value of a floating x and y
	// fmod(x, y) - floating point remainder of x/y
	//				it's like x % y, but the last one deailng wiht integers only
	//				if you'll try to 5 % 4.1, you'll get error (unfortunately)
	// log(x) - natural logarithm of x value (exists log2, log10, log1p etc. also)
	// pow(x, y) - value of x to the power of y
	// round(x) - rounds value to nearest integer
	// sin(x) - sine of x (x is in radians)
	// sinh(x) - hyperbolic sine of a double value
	// tan(x) - tangent of an angle
	// tanh(x) - hyperbolic tangent of a double value

	cout << abs(-5.1) << endl;		  // returns 5.1
	cout << cbrt(0.027) << endl;	  // returns 0.3
	cout << ceil(1.1) << endl;		  // returns 2
	cout << exp(0) - 1 << endl;		  // returns 0
	cout << fabs(-5.1) << endl;		  // returns 5.1
	cout << fdim(6, 5) << endl;		  // returns 1
	cout << fdim(5.1, 5) << endl;	  // returns 0.1
	cout << fmod(5.1, 5) << endl;	  // returns 0.1
	cout << (5 % 4) << endl;		  // returns 1
	cout << log(exp(2)) << endl;	  // returns 2
	cout << log2(4) << endl;		  // returns 2
	return 0;
}