/*
 * MRCPP, a numerical library based on multiresolution analysis and
 * the multiwavelet basis which provide low-scaling algorithms as well as
 * rigorous error control in numerical computations.
 * Copyright (C) 2021 Stig Rune Jensen, Jonas Juselius, Luca Frediani and contributors.
 *
 * This file is part of MRCPP.
 *
 * MRCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MRCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with MRCPP.  If not, see <https://www.gnu.org/licenses/>.
 *
 * For information on the complete list of contributors to MRCPP, see:
 * <https://mrcpp.readthedocs.io/>
 */

#include "OperatorNode.h"
#include "NodeAllocator.h"

#include "utils/Printer.h"
#include "utils/math_utils.h"

using namespace Eigen;

namespace mrcpp {

void OperatorNode::dealloc() {
    int sIdx = this->serialIx;
    this->serialIx = -1;
    this->parentSerialIx = -1;
    this->childSerialIx = -1;
    this->tree->decrementNodeCount(this->getScale());
    if (this->tree->useAllocator()) {
        this->tree->getNodeAllocator().dealloc(sIdx);
    } else {
        this->freeCoefs();
    }
}

/** Calculate one specific component norm of the OperatorNode.
 *
 * OperatorNorms are defined as matrix 2-norms that are expensive to calculate.
 * Thus we calculate some cheaper upper bounds for this norm for thresholding.
 * First a simple vector norm, then a product of the 1- and infinity-norm. */
double OperatorNode::calcComponentNorm(int i) const {
    int depth = getDepth();
    double prec = getOperTree().getNormPrecision();
    double thrs = std::max(MachinePrec, prec / (8.0 * (1 << depth)));

    VectorXd coef_vec;
    this->getCoefs(coef_vec);

    int kp1 = this->getKp1();
    int kp1_d = this->getKp1_d();
    const VectorXd &comp_vec = coef_vec.segment(i * kp1_d, kp1_d);
    const MatrixXd comp_mat = MatrixXd::Map(comp_vec.data(), kp1, kp1);

    double norm = 0.0;
    double vecNorm = comp_vec.norm();
    if (vecNorm > thrs) {
        double infNorm = math_utils::matrix_norm_inf(comp_mat);
        double oneNorm = math_utils::matrix_norm_1(comp_mat);
        if (std::sqrt(infNorm * oneNorm) > thrs) {
            double twoNorm = math_utils::matrix_norm_2(comp_mat);
            if (twoNorm > thrs) norm = twoNorm;
        }
    }
    return norm;
}

void OperatorNode::createChildrenNoBank(bool coefs) {
    if (this->isBranchNode()) MSG_ABORT("Node already has children");

    int nChildren = this->getTDim();
    for (int cIdx = 0; cIdx < nChildren; cIdx++) {
        auto child_p = new OperatorNode(this, cIdx);
        if (coefs) child_p->allocCoefs(this->getTDim(), this->getKp1_d());
        child_p->zeroCoefs();
        child_p->setIsLeafNode();
        child_p->setIsEndNode();
        this->getMWTree().incrementNodeCount(child_p->getScale());
        this->children[cIdx] = child_p;
    }
    this->setIsBranchNode();
    this->clearIsEndNode();
}

void OperatorNode::createChildrenBank(bool coefs) {
    if (this->isBranchNode()) MSG_ABORT("Node already has children");
    auto &allocator = this->getOperTree().getNodeAllocator();

    int nChildren = this->getTDim();
    int sIdx = allocator.alloc(nChildren);

    auto n_coefs = allocator.getNCoefs();
    auto *coefs_p = allocator.getCoef_p(sIdx);
    auto *child_p = allocator.getNode_p(sIdx);

    this->childSerialIx = sIdx;
    for (int cIdx = 0; cIdx < nChildren; cIdx++) {
        // construct into allocator memory
        new (child_p) OperatorNode(this, cIdx);
        this->children[cIdx] = child_p;

        child_p->serialIx = sIdx;
        child_p->parentSerialIx = this->serialIx;
        child_p->childSerialIx = -1;

        child_p->n_coefs = n_coefs;
        child_p->coefs = coefs_p;
        if (coefs) child_p->setIsAllocated();

        child_p->setIsLeafNode();
        child_p->setIsEndNode();
        child_p->clearHasCoefs();

        this->getMWTree().incrementNodeCount(child_p->getScale());
        sIdx++;
        child_p++;
        if (coefs) coefs_p += n_coefs;
    }
    this->setIsBranchNode();
    this->clearIsEndNode();
}

void OperatorNode::genChildrenNoBank() {
    this->createChildrenNoBank(true);
    this->giveChildrenCoefsNoBank();
}

void OperatorNode::genChildrenBank() {
    this->createChildren(true);
    this->giveChildrenCoefs();
}

void OperatorNode::deleteChildren() {
    MWNode<2>::deleteChildren();
    this->setIsEndNode();
}

} // namespace mrcpp
