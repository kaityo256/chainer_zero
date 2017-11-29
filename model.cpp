//------------------------------------------------------------------------
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include "model.hpp"
//------------------------------------------------------------------------
void
import_test(Model &model) {
  vf x(model.n_in, 0.5);
  vf y = model.predict(x);
  printf("[%.8f,%.8f]\n", y[0], y[1]);
}
//------------------------------------------------------------------------
std::vector<vf>
read_file(const std::string filename) {
  std::ifstream ifs(filename);
  std::string str;
  int sum = 0;
  std::vector<vf> vvf;
  while (getline(ifs, str)) {
    std::string token;
    std::istringstream ss(str);
    vf x;
    while (getline(ss, token, ',')) {
      float v = stof(token);
      x.push_back(v);
    }
    vvf.push_back(x);
  }
  return vvf;
}
//------------------------------------------------------------------------
void
file_test(Model &model) {
  std::vector<vf> vvf = read_file("on_test.txt");
  int sum = vvf.size();
  int s = 0;
  for (auto &x : vvf) {
    if (model.argmax(x) == 1)s++;
  }
  vvf = read_file("off_test.txt");
  sum += vvf.size();
  for (auto &x : vvf) {
    if (model.argmax(x) == 0)s++;
  }
  printf("Success/Total = %d/%d\n", s, sum);
  printf("Ratio = %f\n", static_cast<double>(s) / static_cast<double>(sum));
}
//------------------------------------------------------------------------
int
main(void) {
  const int n_in = 10;
  const int n_units = 10;
  const int n_out = 2;
  Model model(n_in, n_units, n_out);
  model.load("test.dat");
  import_test(model);
  file_test(model);
}
//------------------------------------------------------------------------
