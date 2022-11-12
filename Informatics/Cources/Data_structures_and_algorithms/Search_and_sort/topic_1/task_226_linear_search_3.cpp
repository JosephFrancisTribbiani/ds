#include <iostream>
#include <fstream>
#include <vector>


std::vector<int> getIndexes(std::vector<int> nums, int x, int n){
  std::vector<int> indexes;
  for (int i = 0; i < n; i++) {
    if (nums[i] == x) indexes.push_back(i + 1);
  }
  return indexes;
}


int main() {
  int n, x, v_num;
  std::vector<int> nums, indexes;

  if (false) {
    std::ifstream inputfile;
    inputfile.open("./task_226_input.txt", std::ios::in);

    if (!inputfile.is_open()) return EXIT_FAILURE;

    inputfile >> n;
    for (int i = 0; i < n; i++){
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

  indexes = getIndexes(nums, x, n);
  for (int elem : indexes) {
    std::cout << elem << std::endl;
  }

  return 0;
}
