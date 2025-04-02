#ifndef RUN_TESTS_H
#define RUN_TESTS_H

#include <string>
#include <map>
#include <functional>
#include <iostream>

inline void run_tests(const std::string& function_name, const std::map<std::string, std::function<void()>>& fun_handles) {
    if (function_name == "all") {
        std::cout << "Running all registered functions:\n";
        for (const auto& [name, func] : fun_handles) {
            std::cout << "Running: " << name << std::endl;
            func();
        }
    } else {
        auto it = fun_handles.find(function_name);
        if (it != fun_handles.end()) {
            std::cout << "Running: " << function_name << std::endl;
            it->second();
        } else {
            std::cerr << "Function not found: " << function_name << std::endl;
            std::cerr << "Available functions:\n";
            for (const auto& [name, _] : fun_handles) {
                std::cerr << "  " << name << std::endl;
            }
        }
    }
}

#endif // RUN_TESTS_H
