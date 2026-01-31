# -*- coding: utf-8 -*-
"""
Question 2 Analysis Package
两种投票方法对比分析
"""

from .voting_methods import RankMethod, PercentMethod, VotingMethodComparator
from .controversy_analysis import ControversyAnalyzer
from .method_comparison import MethodComparisonAnalyzer
from .detailed_analysis import (
    DetailedWeeklyAnalysis, 
    JudgeTiebreakerSimulator, 
    ControversyTrajectoryVisualizer,
    run_detailed_analysis
)

__all__ = [
    'RankMethod',
    'PercentMethod', 
    'VotingMethodComparator',
    'ControversyAnalyzer',
    'MethodComparisonAnalyzer'
]
