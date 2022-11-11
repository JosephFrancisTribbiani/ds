#include <iostream>
#include <fstream>
#include <vector>

int count_values(std::vector<int> nums, int n, int x) {
  int s = 0;
  for (int i = 0; i < n; i++) {
    if (nums[i] == x) s++;
  }
  return s;
}


int main() {
  // объявление переменных
  int n, x, s, vector_value;
  std::vector<int> nums;

  if (false) {
    std::ifstream inputFile;
    inputFile.open ("./task_223_input.txt", std::ios::in);  // открываем файл для чтения, mode:
                                                            // ios::in (можно было mode не
                                                            // указывать, т.к. для ifstream -
                                                            // Input File stream это значение
                                                            // по умолчанию)

    // проверяем, открыт ли файл
    if (!inputFile.is_open()) {
      std::cout << "Failed during reading the input file";
      return EXIT_FAILURE;
    }
    std::cout << "Reading from file" << std::endl;

    // считывание данных
    inputFile >> n;  // количество элементов в массиве
    for (int i = 0; i < n; i++) {
      inputFile >> vector_value;
      nums.push_back(vector_value);  // записываем значения в массив
    }
    inputFile >> x;

    inputFile.close();  // закрываем файл
  } else {
    std::cin >> n;
    for (int i = 0; i < n; i++) {
      std::cin >> vector_value;
      nums.push_back(vector_value);
    }
    std::cin >> x;
  }

  // итерируемся по элементам массива и считаем вхождение числа x в массив
  s = count_values(nums, n, x);
  std::cout << s;

  return 0;
}
