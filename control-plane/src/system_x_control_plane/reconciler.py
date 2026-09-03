from .budgets import RepairBudget
from .invariants import Observation,evaluate
def check(observation:Observation):return evaluate(observation)
def apply(observation:Observation,budget:RepairBudget|None=None):return {"violations":[{"code":v.code,"owner":v.owner,"message":v.message,"repair":v.repair} for v in evaluate(observation)],"attempts":0,"bounded":True,"changed":False}
