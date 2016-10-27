#include "MWOperator.h"
#include "FunctionTree.h"
#include "WaveletAdaptor.h"
#include "OperApplicationCalculator.h"
#include "MultiResolutionAnalysis.h"
#include "Timer.h"

template<int D>
void MWOperator<D>::operator()(FunctionTree<D> &out,
                               FunctionTree<D> &inp,
                               int maxIter) {
    Timer pre_t;
    this->oper.calcBandWidths(this->apply_prec);
    this->adaptor = new WaveletAdaptor<D>(this->apply_prec, MaxScale);
    this->calculator = new OperApplicationCalculator<D>(this->apply_dir,
                                                        this->apply_prec,
                                                        this->oper,
                                                        inp);
    pre_t.stop();

    this->build(out, maxIter);

    Timer post_t;
    this->clearCalculator();
    this->clearAdaptor();
    this->oper.clearBandWidths();
    out.mwTransform(TopDown, false); // add coarse scale contributions
    out.mwTransform(BottomUp);
    out.calcSquareNorm();
    inp.deleteGenerated();
    post_t.stop();

    println(10, "Time pre operator   " << pre_t);
    println(10, "Time post operator  " << post_t);
    println(10, std::endl);
}

template<int D>
void MWOperator<D>::clearOperator() {
    for (int i = 0; i < this->oper.size(); i++) {
        if (this->oper[i] != 0) delete this->oper[i];
    }
    this->oper.clear();
}

template class MWOperator<1>;
template class MWOperator<2>;
template class MWOperator<3>;
