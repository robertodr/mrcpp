#include "MRCPP/MWFunctions"
#include "MRCPP/Printer"
#include "MRCPP/Timer"
#include "MRCPP/utils/math_utils.h"

const auto min_scale = 0;
const auto max_depth = 20;

const auto order = 5;
const auto prec = 1.0e-3;

const auto D = 6;
int main(int argc, char **argv) {
    auto timer = mrcpp::Timer();

    // Initialize printing
    auto printlevel = 0;
    mrcpp::Printer::init(printlevel);
    mrcpp::print::environment(0);

    // Constructing world box
    auto corner = std::array<int, D>{};
    auto boxes = std::array<int, D>{};
    corner.fill(-1);
    boxes.fill(2);
    auto world = mrcpp::BoundingBox<D>(min_scale, corner, boxes);

    // Constructing basis and MRA
    auto basis = mrcpp::InterpolatingBasis(order);
    auto MRA = mrcpp::MultiResolutionAnalysis<D>(world, basis, max_depth);
    MRA.print();

    // Defining analytic function
    const auto beta = 10.0;
    const auto alpha = std::pow(beta / mrcpp::pi, D / 2.0);
    const auto r_0 = mrcpp::Coord<D>{};
    auto f = [alpha, beta, r_0](const mrcpp::Coord<D> &r) -> double {
        auto R = mrcpp::math_utils::calc_distance<D>(r, r_0);
        return alpha * std::exp(-beta * R * R);
    };

    // Projecting function
    mrcpp::FunctionTree<D> f_tree(MRA);
    mrcpp::project<D>(prec, f_tree, f, -1);
    auto integral = f_tree.integrate();

    mrcpp::print::header(0, "Projecting analytic function");
    mrcpp::print::tree(-1, "Projected function", f_tree, timer);
    mrcpp::print::value(0, "Integrated function", integral, "(au)");
    mrcpp::print::footer(0, timer, 2);

    return 0;
}
