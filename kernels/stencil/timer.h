#pragma once

#include <chrono>
#include <cstdio>

struct Timer {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point stop;
};

inline void startTime(Timer *timer) {
    timer->start = std::chrono::steady_clock::now();
}

inline void stopTime(Timer *timer) {
    timer->stop = std::chrono::steady_clock::now();
}

inline void printElapsedTime(const Timer &timer, const char *label) {
    const auto elapsed = std::chrono::duration<double, std::milli>(
        timer.stop - timer.start);
    std::printf("%s: %.3f ms\n", label, elapsed.count());
}

inline void printElapsedTime(const Timer &timer, const char *label,
                             const char *) {
    printElapsedTime(timer, label);
}
