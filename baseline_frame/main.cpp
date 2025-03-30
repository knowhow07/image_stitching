#include <iostream>
#include <string>
#include "helpers.h"
#include "hw4_challenge1.h"

// Forward declarations from runTests
void run_tests();
void run_all_tests();
void list_functions();
bool run_named_test(const std::string& name);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        list_functions();
        return 0;
    }

    std::string command = argv[1];

    if (command == "all") {
        run_all_tests();
    } else if (!run_named_test(command)) {
        std::cerr << "Unknown command: " << command << std::endl;
        list_functions();
    }

    return 0;
}
