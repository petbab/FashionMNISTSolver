#include <iostream>

int main() {
    std::cout << "Hello, World!\n";

#ifdef DEBUG
    std::cout << "DEBUG\n";
#endif
#ifdef RELEASE
    std::cout << "RELEASE\n";
#endif

    return 0;
}
