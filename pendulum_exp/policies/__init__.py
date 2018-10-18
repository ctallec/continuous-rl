"""Policies subpackage."""
from policies.discrete import AdvantagePolicy
from policies.continuous import AdvantagePolicy as ContinuousAdvantagePolicy

__all__ = ['AdvantagePolicy', 'ContinuousAdvantagePolicy']
