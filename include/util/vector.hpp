#pragma once
#include <iterator>
#include <numeric>
#include <optional>
#include <vector>

template <typename Container, typename Func>
inline auto array_map(Func func, const Container &input) {
    using ResultType = decltype(func(*input.begin()));
    std::vector<ResultType> result;
    result.reserve(std::distance(input.begin(), input.end()));

    for (const auto &item : input) {
        result.push_back(func(item));
    }

    return result;
};

template <typename Container, typename T, typename Compare>
std::optional<typename Container::value_type>
array_binary_find(const Container &container, const T &target, Compare comp);

template <typename Container, typename Reducer, typename T>
T array_reduce(const Container &container, Reducer reducer, T init) {
    return std::accumulate(container.begin(), container.end(), init, reducer);
}

template <std::size_t N1, std::size_t N2>
std::array<double, N1 + N2> array_merge(const std::array<double, N1> &a,
                                        const std::array<double, N2> &b);

template <std::size_t N1, std::size_t N2, std::size_t... Ns>
auto array_merge(const std::array<double, N1> &a,
                 const std::array<double, N2> &b,
                 const std::array<double, Ns> &...rest);
