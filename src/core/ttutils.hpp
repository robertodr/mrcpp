#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <functional>
#include <numeric>
#include <type_traits>

template <std::size_t N> struct RowMajor final {
    inline std::size_t operator()(const std::array<std::size_t, N> &idxs) { return std::inner_product(idxs.cbegin(), idxs.cend(), coefs.cbegin(), 0); }

    RowMajor(const std::array<std::size_t, N> &dims) {
        coefs.back() = 1;
        std::partial_sum(dims.cbegin() + 1, dims.cend(), coefs.begin(), [](std::size_t a, std::size_t b) -> std::size_t { return a * b; });
    }

    std::array<std::size_t, N> coefs;
};

template <std::size_t N> struct ColMajor final {
    inline std::size_t operator()(const std::array<std::size_t, N> &idxs) { return std::inner_product(idxs.cbegin(), idxs.cend(), coefs.cbegin(), 0); }

    ColMajor(const std::array<std::size_t, N> &dims) {
        coefs.front() = 1;
        std::partial_sum(dims.cbegin(), dims.cend() - 1, coefs.begin() + 1, [](std::size_t a, std::size_t b) -> std::size_t { return a * b; });
    }

    std::array<std::size_t, N> coefs;
};

template <typename T> auto to_GiB(std::size_t count) -> double {
    return (count * sizeof(T) / (1024.0 * 1024.0 * 1024.0));
}

/** Compute Frobenius norm of (multidimensional) array.
 *
 * @tparam T scalar type of vector.
 * @param[in] v contiguous array, possibly multidimensional.
 * @param[in] count number of elements in array.
 */
template <typename T> auto frobenius_norm(const T *v, std::size_t count) -> T {
    static_assert(std::is_floating_point_v<T>, "Frobenius norm can only be computed with floating point types");
    return std::sqrt(std::transform_reduce(std::execution::par_unseq, v, v + count, T{0}, std::plus<T>(), [](auto x) { return std::pow(x, 2); }));
}
