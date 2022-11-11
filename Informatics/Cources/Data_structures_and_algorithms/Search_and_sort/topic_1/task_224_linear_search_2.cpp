#include <iostream>
#include <fstream>
#include <vector>


std::string check(std::vector<int> nums, int n, int x) {
  std::string res = "NO";
  for (int i = 0; i < n; i++) {
    if (nums[i] == x) {
      res = "YES";
      break;
    }
  }
  return res;
}


int main() {
  int n, x, v_num;
  std::vector<int> nums;
  std::string res;

  if (false) {
    std::ifstream inputfile;
    inputfile.open("./task_224_input.txt", std::ios::in);

    if (!inputfile.is_open()) {
      std::cout << "Failed during reading the input file";
      return EXIT_FAILURE;
    }

    inputfile >> n;
    for (int i = 0; i < n; i++) {
      inputfile >> v_num;
      nums.push_back(v_num);
    }
    inputfile >> x;

    inputfile.close();
  } else {
    std::cin >> n;
    for (int i = 0; i < n; i++) {
      std::cin >> v_num;
      nums.push_back(v_num);
    }
    std::cin >> x;
  }
  res = check(nums, n, x);
  std::cout << res;

  return 0;
}
