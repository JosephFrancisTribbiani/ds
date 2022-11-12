#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>


int getNearest(std::vector<int> nums, int x, int n) {
  int nearestValue = nums[0];
  for (int i = 1; i < n; i++) {
    if (std::abs(nums[i] - x) < std::abs(nearestValue - x)) nearestValue = nums[i];
  }
  return nearestValue;
}


int main() {
  int n, v_num, x, res;
  std::vector<int> nums;

  if (false) {
    std::ifstream inputfile;
    inputfile.open("./task_225_input.txt", std::ios::in);
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

  } else {
    std::cin >> n;
    for (int i = 0; i < n; i++) {
      std::cin >> v_num;
      nums.push_back(v_num);
    }
    std::cin >> x;
  }
  res = getNearest(nums, x, n);
  std::cout << res;

  return 0;
}
