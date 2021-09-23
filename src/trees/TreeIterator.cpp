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

#include "TreeIterator.h"
#include "HilbertPath.h"
#include "MWNode.h"
#include "utils/Printer.h"

namespace mrcpp {

template <int D, Traverse T, Iterator I>
TreeIterator<D, T, I>::TreeIterator(MWTree<D> &tree)
        : root(0)
        , nRoots(0)
        , maxDepth(-1)
        , state(nullptr)
        , initialState(nullptr) {
    init(tree);
}

template <int D, Traverse T, Iterator I> TreeIterator<D, T, I>::~TreeIterator() {
    if (this->initialState != nullptr) delete this->initialState;
}

template <int D, Traverse T, Iterator I> int TreeIterator<D, T, I>::getChildIndex(int i) const {
    // Legesgue type returns i, Hilbert type returns Hilbert index
    if constexpr (D <= 3 && type == Hilbert) {
        const auto &h = HilbertPath<D>();
        return h.getZIndex(i);
    } else {
        return i;
    }
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::next() {
    if (not this->state) return false;
    if constexpr (mode == TopDown) {
        if (this->tryNode()) return true;
    }
    MWNode<D> &node = *this->state->node;
    if (checkDepth(node) and checkGenerated(node)) {
        const int nChildren = 1 << D;
        for (int i = 0; i < nChildren; i++) {
            int cIdx = getChildIndex(i);
            if (this->tryChild(cIdx)) return true;
        }
    }
    if (this->tryNextRoot()) return true;
    if constexpr (mode == BottomUp) {
        if (this->tryNode()) return true;
    }
    this->removeState();
    return next();
}
template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::nextParent() {
    if (not this->state) return false;
    if constexpr (mode == BottomUp) {
        if (this->tryNode()) return true;
    }
    MWNode<D> &node = *this->state->node;
    if (this->tryNextRootParent()) return true;
    if (checkDepth(node)) {
        if (this->tryParent()) return true;
    }
    if constexpr (mode == TopDown) {
        if (this->tryNode()) return true;
    }
    this->removeState();
    return nextParent();
}

template <int D, Traverse T, Iterator I> void TreeIterator<D, T, I>::init(MWTree<D> &tree) {
    this->root = 0;
    this->maxDepth = -1;
    this->nRoots = tree.getRootBox().size();
    this->state = new IteratorNode<D>(&tree.getRootBox().getNode(this->root));
    // Save the first state so it can be properly deleted later
    this->initialState = this->state;
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::tryNode() {
    if (not this->state) { return false; }
    if (this->state->doneNode) { return false; }
    this->state->doneNode = true;
    return true;
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::tryChild(int i) {
    if (not this->state) { return false; }
    if (this->state->doneChild[i]) { return false; }
    this->state->doneChild[i] = true;
    if (this->state->node->isLeafNode()) { return false; }
    MWNode<D> *child = &this->state->node->getMWChild(i);
    this->state = new IteratorNode<D>(child, this->state);
    return next();
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::tryParent() {
    if (not this->state) return false;
    if (this->state->doneParent) return false;
    this->state->doneParent = true;
    if (not this->state->node->hasParent()) return false;
    MWNode<D> *parent = &this->state->node->getMWParent();
    this->state = new IteratorNode<D>(parent, this->state);
    return nextParent();
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::tryNextRoot() {
    if (not this->state) { return false; }
    if (not this->state->node->isRootNode()) { return false; }
    this->root++;
    if (this->root >= this->nRoots) { return false; }
    MWNode<D> *nextRoot = &state->node->getMWTree().getRootBox().getNode(root);
    this->state = new IteratorNode<D>(nextRoot, this->state);
    return next();
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::tryNextRootParent() {
    if (not this->state) { return false; }
    if (not this->state->node->isRootNode()) { return false; }
    this->root++;
    if (this->root >= this->nRoots) { return false; }
    MWNode<D> *nextRoot = &state->node->getMWTree().getRootBox().getNode(root);
    this->state = new IteratorNode<D>(nextRoot, this->state);
    return nextParent();
}

template <int D, Traverse T, Iterator I> void TreeIterator<D, T, I>::removeState() {
    if (this->state == this->initialState) { this->initialState = nullptr; }
    if (this->state != nullptr) {
        IteratorNode<D> *spare = this->state;
        this->state = spare->next;
        spare->next = nullptr;
        delete spare;
    }
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::checkDepth(const MWNode<D> &node) const {
    if (this->maxDepth < 0) {
        return true;
    } else if (node.getDepth() < this->maxDepth) {
        return true;
    } else {
        return false;
    }
}

template <int D, Traverse T, Iterator I> bool TreeIterator<D, T, I>::checkGenerated(const MWNode<D> &node) const {
    if (node.isEndNode() and not this->returnGenNodes) {
        return false;
    } else {
        return true;
    }
}

template <int D>
IteratorNode<D>::IteratorNode(MWNode<D> *nd, IteratorNode<D> *nx)
        : node(nd)
        , next(nx)
        , doneNode(false)
        , doneParent(false) {
    int nChildren = 1 << D;
    for (int i = 0; i < nChildren; i++) { this->doneChild[i] = false; }
}

template class TreeIterator<1, TopDown, Hilbert>;
template class TreeIterator<2, TopDown, Hilbert>;
template class TreeIterator<3, TopDown, Hilbert>;

template class TreeIterator<1, TopDown, Lebesgue>;
template class TreeIterator<2, TopDown, Lebesgue>;
template class TreeIterator<3, TopDown, Lebesgue>;

template class TreeIterator<1, BottomUp, Hilbert>;
template class TreeIterator<2, BottomUp, Hilbert>;
template class TreeIterator<3, BottomUp, Hilbert>;

template class TreeIterator<1, BottomUp, Lebesgue>;
template class TreeIterator<2, BottomUp, Lebesgue>;
template class TreeIterator<3, BottomUp, Lebesgue>;

template class TreeIterator<6, TopDown, Lebesgue>;
template class TreeIterator<6, BottomUp, Lebesgue>;
} // namespace mrcpp
