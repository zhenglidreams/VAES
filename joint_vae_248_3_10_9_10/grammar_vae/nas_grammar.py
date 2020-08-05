import sys
if '../' not in sys.path:
    sys.path.append('../')
import random
import numpy as np
import nltk
from nltk.parse.generate import generate
from settings import settings as stgs


random.seed(0)  # fix mask and map generation

# gram = """
# S -> 'I' '/' LAY
# LAY -> ND AGG OP '/' LAY | ND AGG OP '/' 'F'
# ND -> '-' | '0' | '2' | '3'
# AGG -> '-'|'0'|'1'
# OP -> CV | SCV | PL | 'D'
# CV -> 'C' CKS BN ACT
# SCV -> 'S' CKS BN ACT
# PL -> 'P' MXA PKS CHF
# CKS -> '0'|'1'|'2'|'3'
# BN -> '0'|'1'
# ACT -> '0'|'1'
# MXA -> '0'|'1'
# PKS -> '0'|'1'
# CHF -> '0'|'1'|'2'|'3'
# Nothing -> None
# """
cfg_str = """
S -> 'I' '/' LAY
LAY -> '-' OP '/' LAY | '-' OP '/' 'F'
OP -> CV | SCV | GCV | PL | 'D'
CV -> 'C' CKS BN ACT
SCV -> 'S' CKS BN ACT
GCV -> 'G' CKS BN ACT
PL -> 'P' MXA PKS CHF
CKS -> '0'|'1'|'2'|'3'
BN -> '0'|'1'
ACT -> '0'|'1'
MXA -> '0'|'1'
PKS -> '0'|'1'
CHF -> '0'|'1'|'2'|'3'
Nothing -> None
"""
# Simple search space, without skip connections (can be strictly context-free):
# cfg_str = """
# S -> 'I' '/' LAY
# LAY -> OP '/' LAY | OP '/' 'F'
# OP -> CV | SCV | GCV | PL | 'D'
# CV -> 'C' CKS BN ACT
# SCV -> 'S' CKS BN ACT
# GCV -> 'G' CKS BN ACT
# PL -> 'P' MXA PKS CHF
# CKS -> '0'|'1'|'2'|'3'
# BN -> '0'|'1'
# ACT -> '0'|'1'
# MXA -> '0'|'1'
# PKS -> '0'|'1'
# CHF -> '0'|'1'|'2'|'3'
# Nothing -> None
# """

# Complex search space, without skip connections:
# cfg_str = """
# S -> 'I' '/' LAY
# LAY -> OP '/' LAY | OP '/' 'F'
# OP -> 'Z' | RN | RNB | DN | DNB | IA | IB | PL
# RN -> 'R' RKS DWN
# RNB -> 'S' RKS DWN
# DN -> 'D' GRW TRA
# DNB -> 'E' GRW TRA
# IA -> 'J' IKS BTL
# IB -> 'K' IKS BTL
# PL -> 'P' MXA PKS
# RKS -> '0' | '1'
# DWN -> '0' | '1'
# GRW -> '0' | '1' | '2'
# TRA -> '0' | '1'
# IKS -> '0' | '1'
# BTL -> '0' | '1' | '2'
# MXA -> '0' | '1'
# PKS -> '0' | '1'
# Nothing -> None
# """

# Minimal grammar for testing:
# cfg_str = """
# S -> 'I' '/' LAY
# LAY -> CV '/' LAY | PL '/' LAY | CV '/' 'F'
# CV -> 'C' '0'
# PL -> 'P' '0'
# Nothing -> None
# # """


class Grammar():
    def __init__(self, cfg_str: str):
        self.GCFG = nltk.CFG.fromstring(cfg_str)
        self.start_index = self.GCFG.productions()[0].lhs()
        self.lhs_list = self._make_lhs_list()
        self.lhs_map = self._make_lhs_map()
        self.rhs_map = self._make_rhs_map()
        self.dim = len(self.rhs_map)
        self.masks = self._make_masks()
        self.ind_of_ind = self._make_ind_of_ind()
        self.max_rhs = max([len(l) for l in self.rhs_map])  # max len of production RHS

    def _make_lhs_list(self):
        all_lhs = [a.lhs().symbol() for a in self.GCFG.productions()]
        return list(set(all_lhs))

    def _make_lhs_map(self):
        lhs_map = {}
        for i, lhs in enumerate(self.lhs_list):
            lhs_map[lhs] = i
        return lhs_map

    def _make_rhs_map(self):
        # Map of (non-terminal) RHS symbol indices for each production rule
        rhs_map = []
        for i, a in enumerate(self.GCFG.productions()):
            rhs_map.append([])
            for b in a.rhs():
                if not isinstance(b, str):
                    s = b.symbol()
                    rhs_map[i].extend(list(np.where(np.array(self.lhs_list) == s)[0]))
        return rhs_map

    def _make_masks(self):
        # Mask indicating which production rules should be masked for each lhs symbol
        masks = np.zeros((len(self.lhs_list), self.dim))
        all_lhs = [a.lhs().symbol() for a in self.GCFG.productions()]
        for i, sym in enumerate(self.lhs_list):
            is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1, -1)
            masks[i] = is_in
        return masks

    def _make_ind_of_ind(self):
        # indices where masks are equal to 1
        index_array = []
        for i in range(self.masks.shape[1]):
            index_array.append(np.where(self.masks[:, i] == 1)[0][0])
        return np.array(index_array)


grammar = Grammar(cfg_str)


if __name__ == '__main__':

    print(grammar.lhs_list)
    print(grammar.lhs_map)
    print(grammar.masks)
    print(grammar.ind_of_ind)