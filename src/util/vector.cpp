#include <algorithm> // std::copy
#include <array>
#include <optional>

template <std::size_t N1, std::size_t N2>
std::array<double, N1 + N2> array_merge(const std::array<double, N1> &a,
                                        const std::array<double, N2> &b) {
    std::array<double, N1 + N2> result{};
    std::copy(a.begin(), a.end(), result.begin());
    std::copy(b.begin(), b.end(), result.begin() + N1);
    return result;
}

// Variadic template for merging more than two arrays
template <std::size_t N1, std::size_t N2, std::size_t... Ns>
auto array_merge(const std::array<double, N1> &a,
                 const std::array<double, N2> &b,
                 const std::array<double, Ns> &...rest) {
    auto first_merge = merge_arrays(a, b);
    return merge_arrays(first_merge, rest...);
}

template <typename Container, typename T, typename Compare>
std::optional<typename Container::value_type>
array_binary_find(const Container &container, const T &target, Compare comp) {
    size_t left = 0;
    size_t right = container.size();

    while (left < right) {
        size_t mid = left + (right - left) / 2;
        auto order = comp(container[mid], target);

        if (order == std::strong_ordering::equal) {
            return container[mid];
        } else if (order == std::strong_ordering::less) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return std::nullopt;
}
