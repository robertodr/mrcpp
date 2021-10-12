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
#include "MWNode.h"
#include "utils/Printer.h"

namespace mrcpp {

template <int D> TreeIterator<D>::TreeIterator(int traverse, int iterator)
        : root(0)
        , nRoots(0)
        , mode(traverse)
        , type(iterator)
        , maxDepth(-1)
        , state(nullptr)
        , initialState(nullptr) {}

template <int D> TreeIterator<D>::TreeIterator(MWTree<D> &tree, int traverse, int iterator)
        : root(0)
        , nRoots(0)
        , mode(traverse)
        , type(iterator)
        , maxDepth(-1)
        , state(nullptr)
        , initialState(nullptr) {
    init(tree);
}

template <int D> TreeIterator<D>::~TreeIterator() {
    if (this->initialState != nullptr) delete this->initialState;
}

template<int D> int TreeIterator<D>::getChildIndex(int i) const {
    const MWNode<D> &node = *this->state->node;
    const HilbertPath<D> &h = node.getHilbertPath();
    // Legesgue type returns i, Hilbert type returns Hilbert index
    return (this->type == Hilbert) ? h.getZIndex(i) : i;
}

template <int D> bool TreeIterator<D>::next() {
    if (not this->state) return false;
    if (this->mode == TopDown) {
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
    if (this->mode == BottomUp) {
        if (this->tryNode()) return true;
    }
    this->removeState();
    return next();
}
template <int D> bool TreeIterator<D>::nextParent() {
    if (not this->state) return false;
    if (this->mode == BottomUp) {
        if (this->tryNode()) return true;
    }
    MWNode<D> &node = *this->state->node;
    if (this->tryNextRootParent()) return true;
    if (checkDepth(node)) {
        if (this->tryParent()) return true;
    }
    if (this->mode == TopDown) {
        if (this->tryNode()) return true;
    }
    this->removeState();
    return nextParent();
}

template <int D> void TreeIterator<D>::init(MWTree<D> &tree) {
    this->root = 0;
    this->maxDepth = -1;
    this->nRoots = tree.getRootBox().size();
    this->state = new IteratorNode<D>(&tree.getRootBox().getNode(this->root));
    // Save the first state so it can be properly deleted later
    this->initialState = this->state;
}

template <int D> bool TreeIterator<D>::tryNode() {
    if (not this->state) { return false; }
    if (this->state->doneNode) { return false; }
    this->state->doneNode = true;
    return true;
}

template <int D> bool TreeIterator<D>::tryChild(int i) {
    if (not this->state) { return false; }
    if (this->state->doneChild[i]) { return false; }
    this->state->doneChild[i] = true;
    if (this->state->node->isLeafNode()) { return false; }
    MWNode<D> *child = &this->state->node->getMWChild(i);
    this->state = new IteratorNode<D>(child, this->state);
    return next();
}

template <int D> bool TreeIterator<D>::tryParent() {
    if (not this->state) return false;
    if (this->state->doneParent) return false;
    this->state->doneParent = true;
    if (not this->state->node->hasParent()) return false;
    MWNode<D> *parent = &this->state->node->getMWParent();
    this->state = new IteratorNode<D>(parent, this->state);
    return nextParent();
}

template <int D> bool TreeIterator<D>::tryNextRoot() {
    if (not this->state) { return false; }
    if (not this->state->node->isRootNode()) { return false; }
    this->root++;
    if (this->root >= this->nRoots) { return false; }
    MWNode<D> *nextRoot = &state->node->getMWTree().getRootBox().getNode(root);
    this->state = new IteratorNode<D>(nextRoot, this->state);
    return next();
}

template <int D> bool TreeIterator<D>::tryNextRootParent() {
    if (not this->state) { return false; }
    if (not this->state->node->isRootNode()) { return false; }
    this->root++;
    if (this->root >= this->nRoots) { return false; }
    MWNode<D> *nextRoot = &state->node->getMWTree().getRootBox().getNode(root);
    this->state = new IteratorNode<D>(nextRoot, this->state);
    return nextParent();
}

template <int D> void TreeIterator<D>::removeState() {
    if (this->state == this->initialState) { this->initialState = nullptr; }
    if (this->state != nullptr) {
        IteratorNode<D> *spare = this->state;
        this->state = spare->next;
        spare->next = nullptr;
        delete spare;
    }
}

template <int D> void TreeIterator<D>::setTraverse(int traverse) {
    switch (traverse) {
        case TopDown:
            this->mode = TopDown;
            break;
        case BottomUp:
            this->mode = BottomUp;
            break;
        default:
            MSG_ABORT("Invalid traverse direction!");
            break;
    }
}

template <int D> void TreeIterator<D>::setIterator(int iterator) {
    switch (iterator) {
        case Lebesgue:
            this->type = Lebesgue;
            break;
        case Hilbert:
            this->type = Hilbert;
            break;
        default:
            MSG_ABORT("Invalid iterator type!");
            break;
    }
}

template <int D> bool TreeIterator<D>::checkDepth(const MWNode<D> &node) const {
    if (this->maxDepth < 0) {
        return true;
    } else if (node.getDepth() < this->maxDepth) {
        return true;
    } else {
        return false;
    }
}

template <int D> bool TreeIterator<D>::checkGenerated(const MWNode<D> &node) const {
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

template class TreeIterator<1>;
template class TreeIterator<2>;
template class TreeIterator<3>;

template class TreeIterator<6>;
} // namespace mrcpp
