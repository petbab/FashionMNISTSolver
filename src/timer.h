#ifndef PV021_TIMER_H
#define PV021_TIMER_H

#include <chrono>
#include <string>
#include <ratio>


class timer {
public:
    explicit timer(std::ostream& out, std::string name = "timer")
        : out{out},
          name{std::move(name)},
          start{std::chrono::steady_clock::now()} {}

    ~timer() {
        auto duration = std::chrono::steady_clock::now() - start;
        auto minutes = std::chrono::floor<std::chrono::minutes>(duration);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration - minutes);
        out << name << ": " << minutes << ' ' << seconds << '\n';
    }

private:
    std::ostream& out;
    std::string name;
    const std::chrono::time_point<std::chrono::steady_clock> start;
};

#endif //PV021_TIMER_H
