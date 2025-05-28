from multivalue import Dialects
southern_am = Dialects.SoutheastAmericanEnclaveDialect()
print(southern_am)
print(southern_am.transform("I talked with them yesterday"))
print(southern_am.executed_rules)