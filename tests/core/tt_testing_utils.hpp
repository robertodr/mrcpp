#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

template <typename DerivedA, typename DerivedB>
inline bool allclose(const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b, const typename DerivedA::RealScalar &rtol = 1e-5, const typename DerivedA::RealScalar &atol = 1e-8) {
    return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

template <typename T, int D> inline auto allclose(const Eigen::Tensor<T, D> &a, const Eigen::Tensor<T, D> &b, T rtol = 1e-5, T atol = 1e-8) -> bool {
    Eigen::Tensor<bool, 0> tmp = ((a - b).abs() <= (atol + rtol * b.abs())).all();
    return tmp.coeff();
}

namespace detail {
inline auto value(int i0, int i1, int i2, int i3, int i4, int i5) -> double {
    return ((i0 / M_2_SQRTPI) - M_E * (i1 * 0.002) + std::pow(i2 + 1, -2) * M_PI * 0.03 - std::sin(i3 * M_PI_4) - 0.005 * std::cos(i5 / (i4 + 1)));
};
} // namespace detail

template <std::size_t... Ds, template <std::size_t> class Indexing = ColMajor, std::size_t N = sizeof...(Ds)> auto sample_tensor() -> Eigen::Tensor<double, N> {
    auto count = (1 * ... * Ds);

    auto dimensions = std::array<std::size_t, N>{Ds...};
    Indexing<N> linear_id(dimensions);

    auto buffer = static_cast<double *>(std::aligned_alloc(alignof(double), sizeof(double) * count));

    // FIXME generalize! this is only really valid for 6-mode tensors!
    // fill
    for (std::size_t i0 = 0; i0 < dimensions[0]; ++i0) {
        for (std::size_t i1 = 0; i1 < dimensions[1]; ++i1) {
            for (std::size_t i2 = 0; i2 < dimensions[2]; ++i2) {
                for (std::size_t i3 = 0; i3 < dimensions[3]; ++i3) {
                    for (std::size_t i4 = 0; i4 < dimensions[4]; ++i4) {
                        for (std::size_t i5 = 0; i5 < dimensions[5]; ++i5) {
                            auto idx = linear_id({i0, i1, i2, i3, i4, i5});
                            buffer[idx] = detail::value(i0, i1, i2, i3, i4, i5);
                        }
                    }
                }
            }
        }
    }

    Eigen::TensorMap<Eigen::Tensor<double, N>> A(buffer, dimensions);

    return A;
}
