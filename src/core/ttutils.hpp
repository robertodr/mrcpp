#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <functional>
#include <numeric>
#include <type_traits>
#include <random>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

#include "utils/Printer.h"

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

template <typename Derived>
auto randomized_svd(const Eigen::MatrixBase<Derived> &A,
                    size_t rank,
                    size_t n_oversamples = 0)
    -> std::tuple<
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
        Eigen::Vector<typename Derived::Scalar, Eigen::Dynamic>,
        Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>> {

    using T = typename Derived::Scalar;
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_type = Eigen::Vector<T, Eigen::Dynamic>;

    auto n_samples = (n_oversamples > 0) ? rank + n_oversamples : 2 * rank;

    // stage A: find approximate range of X

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<T> d{T{0}, T{1}};
    auto normal = [&]() { return d(gen); };
    matrix_type O = matrix_type::NullaryExpr(A.cols(), n_samples, normal);
    matrix_type Y = A * O;

    // orthonormalize
    Eigen::HouseholderQR<Eigen::Ref<matrix_type>> qr(Y);
    auto hh = qr.householderQ();
    matrix_type Q = matrix_type::Identity(Y.rows(), Y.rows());
    Q.applyOnTheLeft(hh);

    // stage B: SVD

    matrix_type B = Q.adjoint() * A;
    auto svd = B.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    if (svd.info() != Eigen::Success) {
        MSG_ABORT("SVD decomposition did not succeed!");
    }

    matrix_type U = (Q * svd.matrixU())(Eigen::all, Eigen::seqN(0, rank));
    vector_type Sigma = svd.singularValues().head(rank);
    matrix_type V = svd.matrixV()(Eigen::all, Eigen::seqN(0, rank));

    return {U, Sigma, V};
}
